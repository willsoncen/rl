import os
import xml.etree.ElementTree as ET
from typing import List, Dict

def generate_route_file(
    output_file: str,
    num_vehicles: int,
    start_time: int = 0,
    end_time: int = 3600
) -> None:
    """
    生成车辆路由文件
    Args:
        output_file: 输出文件路径
        num_vehicles: 车辆数量
        start_time: 开始时间
        end_time: 结束时间
    """
    with open(output_file, 'w') as routes:
        print("""<?xml version="1.0" encoding="UTF-8"?>
        <routes>
            <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="15"/>
        """, file=routes)
        
        # 生成随机车辆
        for i in range(num_vehicles):
            depart_time = start_time + (end_time - start_time) * i / num_vehicles
            print(f'    <vehicle id="veh{i}" type="car" depart="{depart_time}">', file=routes)
            print('        <route edges="edge1 edge2 edge3"/>', file=routes)
            print('    </vehicle>', file=routes)
            
        print('</routes>', file=routes)

def parse_sumo_config(config_file: str) -> Dict:
    """
    解析SUMO配置文件
    Args:
        config_file: 配置文件路径
    Returns:
        配置字典
    """
    tree = ET.parse(config_file)
    root = tree.getroot()
    
    config = {}
    
    # 解析输入文件
    input_section = root.find('input')
    if input_section is not None:
        for child in input_section:
            config[child.tag] = child.get('value')
            
    # 解析时间设置
    time_section = root.find('time')
    if time_section is not None:
        config['begin'] = float(time_section.get('begin', 0))
        config['end'] = float(time_section.get('end', 3600))
        config['step-length'] = float(time_section.get('step-length', 1))
        
    return config

def get_network_info(net_file: str) -> Dict:
    """
    获取路网信息
    Args:
        net_file: 路网文件路径
    Returns:
        路网信息字典
    """
    tree = ET.parse(net_file)
    root = tree.getroot()
    
    info = {
        'junctions': [],
        'edges': [],
        'connections': []
    }
    
    # 解析路口
    for junction in root.findall('junction'):
        info['junctions'].append({
            'id': junction.get('id'),
            'type': junction.get('type'),
            'x': float(junction.get('x')),
            'y': float(junction.get('y'))
        })
        
    # 解析路段
    for edge in root.findall('edge'):
        info['edges'].append({
            'id': edge.get('id'),
            'from': edge.get('from'),
            'to': edge.get('to')
        })
        
    # 解析连接
    for connection in root.findall('connection'):
        info['connections'].append({
            'from': connection.get('from'),
            'to': connection.get('to'),
            'fromLane': int(connection.get('fromLane')),
            'toLane': int(connection.get('toLane'))
        })
        
    return info
