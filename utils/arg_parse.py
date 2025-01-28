import argparse

def process_argparse():
    # 建立解析器
    parser = argparse.ArgumentParser(description='示範程式：處理命令列參數')
    
    # 添加參數
    parser.add_argument('input_file', help='輸入檔案路徑')
    parser.add_argument('output_file', help='輸出檔案路徑')
    parser.add_argument('-n', '--number', type=int, default=10,
                       help='指定一個數字（預設：10）')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='顯示詳細資訊')
    parser.add_argument('--format', choices=['png', 'jpg', 'bmp'],
                       default='png', help='指定輸出格式')
    
    # 解析參數
    args = parser.parse_args()
    
    # 使用參數
    print(f"Input file: {args.input_file}")
    print(f"Output file: {args.output_file}")
    print(f"Number: {args.number}")
    print(f"Verbose: {args.verbose}")
    print(f"Format: {args.format}")
