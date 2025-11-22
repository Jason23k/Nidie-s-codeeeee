import yaml
import os

def validate_yaml(yaml_path):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        print(f"✅ YAML 文件 {yaml_path} 语法正确")
        return True
    except Exception as e:
        print(f"❌ YAML 文件 {yaml_path} 语法错误: {e}")
        return False

if __name__ == "__main__":
    config_path = r"D:\TransVOD\TransVOD-master\configs\transvod_ships.yaml"
    validate_yaml(config_path)