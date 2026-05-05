"""
Brain activation table processor for ARneuro.

This module processes brain activation tables from research papers.
Based on brain_activation_table_processor.py.
"""

import os
import re
import json
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from core.logger import get_logger

logger = get_logger(__name__)


class BrainActivationProcessor:
    """
    Process brain activation tables from research papers.
    """
    
    def __init__(self, llm_client=None, config: Optional[Dict] = None):
        """
        Initialize the brain activation processor.
        
        Args:
            llm_client: LLM client instance
            config: Configuration dictionary
        """
        self.config = config or {}
        self.llm_client = llm_client
        
    def set_llm_client(self, llm_client):
        """
        Set the LLM client for processing.
        
        Args:
            llm_client: LLM client instance
        """
        self.llm_client = llm_client
    
    def extract_tables_with_context(self, 
                                   md_text: str, 
                                   context_lines: int = 5) -> List[Dict]:
        """
        Extract all tables from markdown text with context.
        
        Args:
            md_text: Markdown file complete text
            context_lines: Number of context lines before and after table
            
        Returns:
            List[Dict]: List of table information dictionaries
        """
        lines = md_text.split('\n')
        tables_info = []
        
        # Pattern for markdown tables
        table_pattern = r'^\|.*\|$'
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if line starts a markdown table
            if re.match(table_pattern, line.strip()):
                table_start = i
                table_lines = []
                
                # Collect all consecutive table rows
                while i < len(lines) and (re.match(table_pattern, lines[i].strip()) or lines[i].strip() == ''):
                    table_lines.append(lines[i])
                    i += 1
                
                # Join table lines
                table_text = '\n'.join(table_lines)
                
                # Get context before table
                before_start = max(0, table_start - context_lines)
                before_context = '\n'.join(lines[before_start:table_start])
                
                # Get context after table
                after_end = min(len(lines), i + context_lines)
                after_context = '\n'.join(lines[i:after_end])
                
                # Full context
                full_context = f"{before_context}\n[TABLE {len(tables_info) + 1}]\n{after_context}"
                
                # Convert markdown table to HTML for better parsing
                table_html = self._markdown_to_html(table_text)
                
                tables_info.append({
                    'table_index': len(tables_info),
                    'table_html': table_html,
                    'table_text': table_text,
                    'before_context': before_context,
                    'after_context': after_context,
                    'full_context': full_context
                })
            else:
                i += 1
        
        logger.info(f"Extracted {len(tables_info)} tables from markdown text")
        return tables_info
    
    def _markdown_to_html(self, markdown_table: str) -> str:
        """
        Convert markdown table to HTML.
        
        Args:
            markdown_table: Markdown table text
            
        Returns:
            str: HTML table
        """
        lines = markdown_table.strip().split('\n')
        if not lines:
            return ''
        
        html = ['<table>']
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Remove leading/trailing pipes
            if line.startswith('|'):
                line = line[1:]
            if line.endswith('|'):
                line = line[:-1]
            
            # Split by pipe
            cells = [cell.strip() for cell in line.split('|')]
            
            # Determine if this is a separator row
            is_separator = all(re.match(r'^:?-+:?$', cell.strip()) for cell in cells if cell)
            
            if is_separator:
                continue  # Skip separator rows in HTML
            
            html.append('  <tr>')
            for cell in cells:
                if cell:
                    tag = 'th' if i == 0 else 'td'
                    html.append(f'    <{tag}>{cell}</{tag}>')
            html.append('  </tr>')
        
        html.append('</table>')
        return '\n'.join(html)
    
    def assess_brain_coordinates(self, 
                                table_text: str, 
                                table_description: str) -> Dict[str, Any]:
        """
        Assess if a table contains brain coordinates and extract task name.
        
        Args:
            table_text: Table content
            table_description: Table description/context
            
        Returns:
            Dict: Assessment results
        """
        client, model_name = self._require_llm_client()
        
        system_prompt = """
        You are an expert in neuroimaging data analysis. Your task is to analyze a table from a neuroscience paper and determine:

        1. Does this table explicitly report brain activation coordinates (x, y, z values)?
        2. What is the exact task name associated with these coordinates?

        Coordinate detection rules:
        - Look for columns or rows containing x, y, z values (e.g., "x = 12, y = -34, z = 56").
        - Accept also formats like "MNI(X,Y,Z)", "Peak (x,y,z)", or three separate columns named X, Y, Z.
        - Do NOT count values that are clearly not coordinates (e.g., cluster size, t-values, p-values) unless they are accompanied by explicit x/y/z.

        How to extract the task name:
        - Use the exact task label stated in the table or description, such as "n-back", "Stroop", "Go/No-Go", "motor task", "language task", "resting-state", etc.
        - If multiple tasks are named, choose the one that the coordinates are explicitly reported for.
        - If no explicit task name is provided, set it to "None".

        Header detection:
        - If a distinct header row exists, return it as "Table_header" (verbatim text). If no clear header is found, return "None".

        Output format (IMPORTANT):
        Return analysis results as a JSON object with the following structure ONLY:
        {
            "Table_header": "Table header content",
            "reason": "Think process steps",
            "contains_coordinates": "Yes or No",
            "Task_name": "the exact name or None"
        }
        The json structural must be EXACTLY the same as above. Do not change key names, especially the name: contains_coordinates. Only return the JSON object, no explanations or additional text.
        """
        
        user_prompt = f"""
        Table content:
        {table_text}

        Table description:
        {table_description}

        Analyze this table to decide if it explicitly reports brain coordinates and extract the exact task name.
        """
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
                max_tokens=8192
            )
            
            result = json.loads(response.choices[0].message.content)
            
            return {
                'contains_coordinates': result.get('contains_coordinates', 'No') == 'Yes',
                'Task_name': result.get('Task_name', None),
                'reason': result.get('reason', ''),
                'Table_header': result.get('Table_header', '')
            }
            
        except Exception as e:
            logger.error(f"LLM assessment failed: {e}")
            raise
    
    def _assess_brain_coordinates_rule_based(self, table_text: str, table_description: str) -> Dict[str, Any]:
        """Deprecated: rule-based assessment is intentionally disabled."""
        raise NotImplementedError("Rule-based brain coordinate assessment is disabled; use LLM-based assessment only")

    def fix_and_parse_table(self, 
                           table_text: str, 
                           table_description: str, 
                           task_name: str) -> pd.DataFrame:
        """
        Fix and parse table to DataFrame.
        
        Args:
            table_text: Table content
            table_description: Table description
            task_name: Task name
            
        Returns:
            pd.DataFrame: Parsed table data
        """
        client, model_name = self._require_llm_client()
        
        system_prompt = """
        You are a professional brain imaging table data processing expert. Parse the table and return a CSV format with proper column names.

        IMPORTANT: You must identify the brain space (MNI or Talairach) from the table description or content. This is REQUIRED.

        Output requirements:
        1. Column names must be clear and descriptive, including:
           - Brain_Region or Anatomical_Region (脑区)
           - X, Y, Z (坐标)
           - Cluster_Size or Number_of_Voxels (聚类大小/体素数)
           - Statistic_Value (统计值，如t值或z值)
           - p_value (p值)
           - Brain_Space (脑空间：MNI 或 Talairach)
           - Task_Name (任务名称)

        2. If the table contains coordinates (x, y, z), you MUST include a Brain_Space column.
           - Look for keywords: "MNI", "Talairach", "MNI152", etc. in the table or description
           - If not specified, infer from coordinate ranges (MNI: x∈[-90,90], y∈[-120,120], z∈[-90,90])
           - Default to "MNI" if unclear

        3. Return ONLY a valid CSV string with proper headers. No explanations.
        """
        
        user_prompt = f"""
        Table content:
        {table_text}

        Table description:
        {table_description}

        Task name: {task_name}

        Parse this table into a clean CSV format with the required columns.
        """
        
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=8192
            )
            
            csv_text = response.choices[0].message.content.strip()
            
            # Clean up CSV text
            csv_text = self._clean_csv_text(csv_text)
            
            # Parse CSV to DataFrame
            try:
                df = pd.read_csv(pd.io.common.StringIO(csv_text))
                logger.info(f"Parsed table with {len(df)} rows and {len(df.columns)} columns")
                return df
            except Exception as e:
                logger.error(f"Failed to parse CSV: {e}")
                logger.debug(f"CSV text: {csv_text}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"LLM table parsing failed: {e}")
            raise
    

    def _require_llm_client(self):
        """Resolve and validate an OpenAI-compatible chat client and model name."""
        if self.llm_client is None:
            raise ValueError("LLM client is required for table processing")

        model_name = self.config.get('model_name', 'deepseek-chat')
        client = self.llm_client

        # Support passing LLMClientManager directly.
        if hasattr(client, 'get_client') and not hasattr(client, 'chat'):
            client_type = self.config.get('llm_client_type', 'deepseek')
            client, resolved_model = client.get_client(client_type=client_type, model_name=model_name)
            model_name = resolved_model or model_name

        if not hasattr(client, 'chat'):
            raise TypeError(f"Invalid LLM client object: {type(client)}; expected OpenAI-compatible client with .chat")

        return client, model_name

    def _clean_csv_text(self, csv_text: str) -> str:
        """
        Clean CSV text from LLM response.
        """
        # Remove markdown code blocks
        csv_text = re.sub(r'```[a-z]*\n', '', csv_text)
        csv_text = re.sub(r'\n```', '', csv_text)
        
        # Remove any explanatory text before/after CSV
        lines = csv_text.split('\n')
        csv_lines = []
        in_csv = False
        
        for line in lines:
            # Check if line looks like CSV (contains commas or is header-like)
            if ',' in line or any(col in line.lower() for col in ['brain', 'region', 'x', 'y', 'z', 'cluster', 'statistic', 'p_value', 'task']):
                in_csv = True
                csv_lines.append(line)
            elif in_csv and line.strip() and not line.strip().startswith('"') and ',' not in line:
                # Might be end of CSV
                break
        
        return '\n'.join(csv_lines)
    
    def _parse_table_basic(self, table_text: str) -> pd.DataFrame:
        """
        Basic parsing of markdown table.
        """
        try:
            # Split into lines
            lines = [line.strip() for line in table_text.split('\n') if line.strip()]
            
            if not lines:
                return pd.DataFrame()
            
            # Find separator row (contains only dashes and colons)
            separator_idx = -1
            for i, line in enumerate(lines):
                if re.match(r'^[\|:\-\s]+$', line.replace('|', '').strip()):
                    separator_idx = i
                    break
            
            # Extract headers
            if separator_idx > 0:
                header_line = lines[separator_idx - 1]
            else:
                header_line = lines[0]
            
            # Parse headers
            headers = [cell.strip() for cell in header_line.split('|') if cell.strip()]
            
            # Extract data rows
            data_start = separator_idx + 1 if separator_idx >= 0 else 1
            data_rows = []
            
            for i in range(data_start, len(lines)):
                if '|' in lines[i]:
                    cells = [cell.strip() for cell in lines[i].split('|') if cell.strip()]
                    if len(cells) == len(headers):
                        data_rows.append(cells)
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=headers)
            logger.info(f"Basic parsing: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Basic table parsing failed: {e}")
            return pd.DataFrame()
    
    def save_table_data(self, 
                       df: pd.DataFrame, 
                       table_info: Dict[str, Any],
                       output_dir: str,
                       filename: Optional[str] = None) -> str:
        """
        Save table data to CSV file.
        
        Args:
            df: DataFrame with table data
            table_info: Table information dictionary
            output_dir: Output directory
            filename: Output filename (optional)
            
        Returns:
            str: Path to saved file
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            table_idx = table_info.get('table_index', 0)
            filename = f"table_{table_idx + 1}_data.csv"
        
        output_path = os.path.join(output_dir, filename)
        
        # Save DataFrame
        df.to_csv(output_path, index=False)
        
        # Save metadata
        metadata = {
            'table_index': table_info.get('table_index'),
            'contains_coordinates': table_info.get('contains_coordinates', False),
            'task_name': table_info.get('Task_name'),
            'table_header': table_info.get('Table_header'),
            'reason': table_info.get('reason'),
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns)
        }
        
        metadata_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved table data to: {output_path}")
        return output_path