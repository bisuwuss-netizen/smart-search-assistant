import os
from dotenv import load_dotenv
load_dotenv()
from src.tools import create_search_tool
def test_create_search_tool():
    query = "2025年诺贝尔物理学奖得主是谁"

    result = create_search_tool().invoke(query)
    print(result)


if __name__ == '__main__':
    print("="*60)
    print("测试工具：")
    print("="*60)
    test_create_search_tool()
