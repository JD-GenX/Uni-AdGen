#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
from typing import List, Tuple, Dict, Any, Optional
import argparse
import os
from datetime import datetime
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('json_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def text_aware_filter(sku_title, history_titles, history_images, embedding_model):
    """
    基于文本相似性计算历史数据的相似度分数

    Args:
        sku_title: SKU标题文本
        history_titles: 历史标题列表
        history_images: 历史图像列表
        embedding_model: 文本嵌入模型

    Returns:
        tuple: (完整的history_titles, 完整的history_images, 相似性分数列表)
            - history_titles: 原始完整的历史标题列表
            - history_images: 原始完整的历史图像列表
            - similarities: 相似性分数列表，与history_titles严格索引对应

    Note:
        similarities[i]对应history_titles[i]的相似性值
    """
    # 分别提取history_titles和sku_title的特征
    all_texts = [sku_title] + history_titles
    text_features = embedding_model.encode(all_texts, show_progress_bar=False)    
    
    # 分离sku_title和history_titles的特征向量
    sku_feature_dense = text_features[:1]  
    history_features_dense = text_features[1:] 

    # 分别计算sku_title和history_titles的相似度
    similarities = cosine_similarity(sku_feature_dense, history_features_dense).flatten()
    
    return history_titles, history_images, similarities.tolist()


class JSONProcessor:
    """JSON文件处理器主类"""
    
    def __init__(self, input_file: str, output_dir: str = "output"):
        """
        初始化处理器
        
        Args:
            input_file: 输入JSON文件路径
            output_dir: 输出目录
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.ensure_output_dir()
    
    def ensure_output_dir(self):
        """确保输出目录存在"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"创建输出目录: {self.output_dir}")
    
    def load_json_data(self) -> List[Dict[str, Any]]:
        """
        加载JSON数据，支持JSONL格式（每行一个JSON对象）
        
        Returns:
            List[Dict[str, Any]]: JSON数据列表
        """
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            
            logger.info(f"成功加载{len(data)}条JSON记录")
            return data
            
        except FileNotFoundError:
            logger.error(f"文件未找到: {self.input_file}")
            raise
        except Exception as e:
            logger.error(f"加载文件时发生错误: {e}")
            raise
    
    def validate_data_structure(self, data: List[Dict[str, Any]]) -> bool:
        """
        验证数据结构完整性
        
        Args:
            data: JSON数据列表
        
        Returns:
            bool: 验证是否通过
        """
        required_fields = ['history_images', 'history_titles']
        
        for i, item in enumerate(data):
            # 检查必需字段是否存在
            for field in required_fields:
                if field not in item:
                    logger.error(f"第{i+1}条记录缺少必需字段: {field}")
                    return False
                
                if not isinstance(item[field], list):
                    logger.error(f"第{i+1}条记录的{field}不是列表类型")
                    return False
            
            # 检查索引对应关系
            if len(item['history_titles']) != len(item['history_images']):
                logger.error(f"第{i+1}条记录的标题数量({len(item['history_titles'])})与图像数量({len(item['history_images'])})不匹配")
                return False
        
        logger.info("数据结构验证通过")
        return True
       
    def process_with_text_filter(self, data: List[Dict[str, Any]], embedding_model,
                                sku_title_field: str = 'sku_title') -> List[Dict[str, Any]]:
        """
        使用文本过滤方式处理数据

        Args:
            data: 原始数据
            embedding_model: 文本嵌入模型，用于计算文本相似性
            sku_title_field: SKU标题字段名

        Returns:
            List[Dict[str, Any]]: 处理后的数据，包含以下新增字段：
                - text_similarity: 相似性数组，与history_titles数组保持严格的索引对应关系
                  即text_similarity[i]对应history_titles[i]的相似性值

        Note:
            - 保留完整的历史数据，不进行截断操作
            - 确保text_similarity数组与完整的history_titles数组索引一一对应
        """
        logger.info(f"开始文本过滤处理 (sku_title_field={sku_title_field})")
        processed_data = []
        success_count = 0
        
        for i, item in enumerate(tqdm(data)):
            try:
                # 创建副本以避免修改原始数据
                processed_item = item.copy()
                
                # 获取SKU标题
                sku_title = item.get(sku_title_field, '')
                if not sku_title:
                    logger.warning(f"第{i+1}条记录缺少SKU标题字段: {sku_title_field}，跳过处理")
                    continue
                
                # 应用文本相似性计算函数
                complete_titles, complete_images, text_similarities = text_aware_filter(
                    sku_title,
                    item['history_titles'],
                    item['history_images'],
                    embedding_model = embedding_model
                )
                
                # 验证返回的完整数据
                if len(complete_titles) != len(complete_images):
                    logger.error(f"第{i+1}条记录标题和图像数量不匹配")
                    continue
                
                # 验证相似性数据与完整历史数据的索引对应关系
                if len(text_similarities) != len(complete_titles):
                    logger.error(f"第{i+1}条记录相似性数据与历史数据数量不匹配")
                    continue

                # 更新处理后的数据 - 保留完整历史数据并添加text_similarity字段
                processed_item['history_titles'] = complete_titles
                processed_item['history_images'] = complete_images
                processed_item['text_similarity'] = text_similarities  # 相似性数据，与history_titles严格索引对应
                
                processed_data.append(processed_item)
                success_count += 1
                
            except Exception as e:
                logger.error(f"处理第{i+1}条记录时发生错误: {e}")
                continue
        
        logger.info(f"文本过滤处理完成，成功处理{success_count}/{len(data)}条记录")
        return processed_data
    
    def save_processed_data(self, data: List[Dict[str, Any]], output_filename: str):
        output_path = os.path.join(self.output_dir, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"数据已保存到: {output_path} (共{len(data)}条记录)")
            
        except Exception as e:
            logger.error(f"保存文件时发生错误: {e}") 


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(
        description='JSON文件处理器 - 支持随机和文本过滤',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('input_file', help='输入JSON文件路径（支持JSONL格式）')
    parser.add_argument('--output', type=str, required=True, help='输出JSON文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='文本嵌入模型权重路径')
    parser.add_argument('--sku_title_field', default='sku_title_cn', help='SKU标题字段名 (默认: sku_title_cn)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='日志级别 (默认: INFO)')
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # 验证输入文件存在
        if not os.path.exists(args.input_file):
            logger.error(f"输入文件不存在: {args.input_file}")
            return 1
        
        # 初始化处理器
        processor = JSONProcessor(args.input_file, args.output)
        
        # 加载数据
        logger.info("开始加载JSON数据...")
        data = processor.load_json_data()
        
        if not data:
            logger.error("没有加载到有效的JSON数据")
            return 1
        
        # 验证数据结构
        logger.info("验证数据结构...")
        if not processor.validate_data_structure(data):
            logger.error("数据结构验证失败，程序终止")
            return 1
        
        text_data = []

        embedding_model = SentenceTransformer(args.checkpoint)
        
        logger.info("执行文本过滤处理...")
        text_data = processor.process_with_text_filter(data, embedding_model, args.sku_title_field)
        if text_data:
            processor.save_processed_data(text_data, 'text_filtered.json')

        
        logger.info("所有处理完成！")
        return 0
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main())