import xml.etree.ElementTree as ET
import argparse
import sys

def add_dependencies(input_file, output_file):
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML file: {e}")
        return

    # 获取 GPU 总数
    ngpus = int(root.get('ngpus', 0))
    if ngpus == 0:
        print("Error: Could not determine ngpus from XML root.")
        return

    print(f"Processing for {ngpus} GPUs...")

    # 遍历所有 GPU
    for gpu in root.findall('gpu'):
        rank = int(gpu.get('id'))
        
        # 1. 计算 Pairwise 算法的 Step 顺序
        # 逻辑：for step in range(1, num_ranks): dst = rank ^ step
        # dest_sequence 存储的是当前 Rank 在每一步需要发送的目标 Rank 列表，顺序即为执行顺序
        dest_sequence = []
        for step in range(1, ngpus):
            dest_rank = rank ^ step
            dest_sequence.append(dest_rank)
        
        # 2. 将当前 GPU 的所有 TB 按照发送目标(send属性)进行分组
        # key: dest_rank, value: list of tb elements
        tbs_by_dest = {d: [] for d in dest_sequence}
        
        for tb in gpu.findall('tb'):
            send_target = int(tb.get('send', -1))
            # 我们只处理作为发送者的 TB (send != -1)
            if send_target in tbs_by_dest:
                tbs_by_dest[send_target].append(tb)
        
        # 3. 确保每个组内的 TB 按照 ID 排序，以保证 Instance 之间的对应关系正确
        # 例如：Step 1 的第 i 个 TB 应该对应 Step 2 的第 i 个 TB
        for d in tbs_by_dest:
            tbs_by_dest[d].sort(key=lambda x: int(x.get('id')))

        # 4. 添加依赖关系链
        # 遍历 Step 顺序，从第2个目标开始，让它依赖于前一个目标
        for i in range(1, len(dest_sequence)):
            prev_dest = dest_sequence[i-1]
            curr_dest = dest_sequence[i]
            
            prev_tb_list = tbs_by_dest[prev_dest]
            curr_tb_list = tbs_by_dest[curr_dest]
            
            # 检查 Instance 数量是否匹配 (通常应该是匹配的，即 instances 参数)
            if len(prev_tb_list) != len(curr_tb_list):
                print(f"Warning: GPU {rank} has mismatched instance counts between send->{prev_dest} and send->{curr_dest}. Skipping dependency chain.")
                continue
            
            # 为每个 Instance 建立依赖
            for prev_tb, curr_tb in zip(prev_tb_list, curr_tb_list):
                # 找到当前 TB 中的执行步骤 (step)
                # XML 中通常只有一个 step，但我们还是遍历查找
                for step_node in curr_tb.findall('step'):
                    # 修改 XML 属性添加依赖
                    # depid: 依赖的 threadblock ID
                    # deps: 依赖的 step index (上一个 TB 的 step 通常是 0)
                    # hasdep: 标记为有依赖
                    step_node.set('depid', prev_tb.get('id'))
                    step_node.set('deps', '0') 
                    step_node.set('hasdep', '1')

    # 保存文件
    tree.write(output_file, encoding='UTF-8', xml_declaration=False)
    print(f"Successfully wrote modified XML to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add serial dependencies to MSCCL pairwise XML.")
    parser.add_argument("input_xml", help="Path to the input XML file (e.g., pairwise.xml)")
    parser.add_argument("output_xml", help="Path to save the output XML file")
    
    args = parser.parse_args()
    
    add_dependencies(args.input_xml, args.output_xml)