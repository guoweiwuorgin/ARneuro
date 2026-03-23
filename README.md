# ARneuro: 基于LLM的文献定量综述工具

ARneuro是一个完整的文献处理流水线，专门用于脑成像和神经科学领域的文献定量综述。

## 功能特性

### 核心功能
1. **PubMed文献元数据获取** - 从CSV文件读取PMID并获取文献信息
2. **PDF下载** - 支持多种期刊格式（ACS, NEJM, Science Direct等）
3. **GLM-OCR处理** - 使用GLM-OCR将PDF转换为Markdown格式
4. **文档结构分割** - 识别和分割文献结构（必需：Methods, Results）
5. **特征提取** - 使用LLM从文献中提取结构化信息
6. **脑激活表格处理** - 提取和分析脑激活坐标表格

### 技术特点
- **模块化设计** - 每个功能独立模块，易于维护和扩展
- **配置驱动** - 所有参数通过YAML配置文件管理
- **错误恢复** - 支持断点续传和失败重试
- **质量评估** - 自动评估OCR质量和处理结果
- **批量处理** - 支持大规模文献处理

## 安装要求

### Python版本
- Python ≥ 3.10

### 系统依赖
- GLM-OCR (需要单独安装)
- CUDA (可选，用于GPU加速)

### Python依赖
```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 安装ARneuro
```bash
# 克隆仓库
git clone <repository-url>
cd ARneuro

# 安装依赖
pip install -r requirements.txt

# 安装GLM-OCR (需要单独安装)
# 请参考GLM-OCR官方文档
```

### 2. 准备输入数据
创建CSV文件，包含以下列（至少需要PMID列）：
```csv
PMID,Title,Authors,Citation,First Author,Journal/Book,Publication Year,Create Date,PMCID,NIHMS ID
12345678,Sample Article,Author A et al.,J Neurosci. 2023,Author A,Journal of Neuroscience,2023,2023-01-01,PMC1234567,NIHMS123456
```

### 3. 运行完整流水线
```python
from arneuro import ARneuroPipeline

# 初始化流水线
pipeline = ARneuroPipeline()

# 运行完整处理
result = pipeline.run_full_pipeline("input.csv")

# 查看结果
print(f"处理完成: {result.successful_papers}成功, {result.failed_papers}失败")
```

### 4. 使用配置文件
创建`config.yaml`文件：
```yaml
pdf_download:
  output_dir: "./data/pdfs"
  max_retries: 3
  timeout: 30
  finders:
    - generic_citation
    - pubmed_central
    - acs
    - nejm

ocr_processing:
  backend: "local"   # 可选 local 或 api
  model_path: "/storage/work/wuguowei/Bigmodel/GLM-OCR"
  device: "cuda"
  batch_size: 4
  output_dir: "./data/markdown"
  glmocr_cli_path: "glmocr"
  language: "ch+en"
  api_key: "${GLM_API_KEY}"
  api_base_url: "https://open.bigmodel.cn/api/paas/v4/layout_parsing"
  api_model: "glm-ocr"
  api_timeout: 300
  api_use_base64: true

paths:
  data_dir: "./data"
  logs_dir: "./logs"
  cache_dir: "./cache"
```

## 模块说明

### 1. PDF下载模块 (`PDFDownloader`)
- 支持8种期刊finder
- 自动重试机制
- 进度跟踪和错误记录

### 2. OCR处理模块 (`GLMOCRProcessor`)
- 集成本地GLM-OCR CLI 与智谱AI在线GLM-OCR API
- 质量评估和自动修复
- 支持中英文混合识别

### 3. 文档分割模块 (待实现)
- 必需部分验证（Methods, Results）
- 灵活的结构识别
- 表格提取和格式化

### 4. 特征提取模块 (待实现)
- LLM驱动的结构化提取
- 任务、人口学、方法信息提取
- 脑激活坐标提取

## 使用示例

### 示例1: 批量处理文献
```python
from arneuro import ARneuroPipeline

# 初始化
pipeline = ARneuroPipeline("config.yaml")

# 运行完整流水线
result = pipeline.run_full_pipeline("literature_list.csv")

# 导出结果
pipeline.export_results("output/")
```

### 示例2: 单独使用PDF下载
```python
from arneuro import PDFDownloader

downloader = PDFDownloader()
result = downloader.download_batch(["12345678", "23456789"])
print(f"下载完成: {result.succeeded}成功, {result.failed}失败")
```

### 示例3: 单独使用OCR处理
```python
from arneuro import GLMOCRProcessor

# 方式1：本地模型
local_processor = GLMOCRProcessor()
local_result = local_processor.process_pdf("data/pdfs/12345678.pdf")

# 方式2：在线API
api_processor = GLMOCRProcessor(backend="api")
api_result = api_processor.process_pdf("data/pdfs/12345678.pdf")
print(f"本地OCR质量: {local_result.quality_level.value} ({local_result.quality_score:.1f}%)")
print(f"API OCR质量: {api_result.quality_level.value} ({api_result.quality_score:.1f}%)")
```

## 配置说明

### 主要配置项

#### PDF下载配置
```yaml
pdf_download:
  output_dir: "./data/pdfs"  # PDF输出目录
  max_retries: 3             # 最大重试次数
  timeout: 30                # 超时时间（秒）
  finders:                   # 使用的查找器
    - generic_citation
    - pubmed_central
    - acs
    - nejm
    - science_direct
```

#### OCR处理配置
```yaml
ocr_processing:
  backend: "local"                # local=本地CLI, api=在线调用
  model_path: "/path/to/glm-ocr"  # GLM-OCR模型路径
  device: "cuda"                  # 设备类型 (cuda/cpu)
  batch_size: 4                    # 批处理大小
  output_dir: "./data/markdown"   # Markdown输出目录
  glmocr_cli_path: "glmocr"       # GLM-OCR命令行路径
  language: "ch+en"               # 识别语言
  api_key: "${GLM_API_KEY}"       # 智谱AI API Key
  api_base_url: "https://open.bigmodel.cn/api/paas/v4/layout_parsing"
  api_model: "glm-ocr"
  api_timeout: 300
  api_use_base64: true             # 将本地PDF转base64后提交给API
```

#### 路径配置
```yaml
paths:
  data_dir: "./data"     # 数据目录
  logs_dir: "./logs"     # 日志目录
  cache_dir: "./cache"   # 缓存目录
  temp_dir: "./temp"     # 临时目录
```

## 输出格式

### 处理报告
处理完成后生成`processing_report.json`，包含：
- 处理统计（成功/失败数量）
- PDF下载统计
- OCR处理统计
- 每篇文献的详细状态

### 成功文献列表
生成`successful_papers.csv`，包含：
- PMID
- PDF路径
- Markdown路径
- OCR质量评分
- 处理时间

### 失败文献列表
生成`failed_papers.csv`，包含：
- PMID
- 失败阶段
- 错误信息
- 处理时间

## 故障排除

### 常见问题

1. **GLM-OCR未找到**
   ```
   错误: GLM-OCR命令未找到
   解决: 确保GLM-OCR已安装并添加到PATH环境变量
   ```

2. **PDF下载失败**
   ```
   错误: 未找到PDF链接
   解决: 检查PMID是否正确，或尝试手动下载
   ```

3. **内存不足**
   ```
   错误: CUDA out of memory
   解决: 减小batch_size，或使用CPU模式
   ```

### 调试模式
设置环境变量启用详细日志：
```bash
export ARNEURO_LOG_LEVEL=DEBUG
```

## 许可证

本项目采用AGPL-3.0许可证。详情请见LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至: <email>

## 致谢

- GLM-OCR团队提供的优秀OCR工具
- 所有贡献者和用户的支持