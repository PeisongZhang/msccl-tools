import xml.etree.ElementTree as ET
import argparse
import sys

def indent(elem, level=0):
    """
    用于美化XML输出的辅助函数
    """
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def add_dependencies(input_file, output_file):
    try:
        tree = ET.parse(input_file)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except ET.ParseError:
        print(f"Error: Failed to parse '{input_file}'.")
        return

    # 获取 GPU 数量
    gpus = root.findall('gpu')
    num_gpus = len(gpus)
    print(f"Detected {num_gpus} GPUs.")

    for gpu in gpus:
        rank = int(gpu.get('id'))
        
        # 根据 linear_shift.py 的逻辑推导发送顺序
        # 逻辑: for step in range(1, num_ranks): dst = (src + step) % num_ranks
        dest_sequence = []
        for step in range(1, num_gpus):
            dst = (rank + step) % num_gpus
            dest_sequence.append(dst)

        # 收集每个目标 Rank 对应的 TB 列表
        # 结构: { dest_rank: [tb_element1, tb_element2, ...], ... }
        tb_groups = {dst: [] for dst in dest_sequence}
        
        for tb in gpu.findall('tb'):
            send_target = int(tb.get('send', -1))
            if send_target in tb_groups:
                tb_groups[send_target].append(tb)

        # 确保每个组内的 TB 是按 ID 排序的 (对应 instance 0, 1, 2...)
        for dst in tb_groups:
            tb_groups[dst].sort(key=lambda x: int(x.get('id')))

        # 添加链式依赖
        # Step k 依赖于 Step k-1
        for i in range(1, len(dest_sequence)):
            prev_dst = dest_sequence[i-1]
            curr_dst = dest_sequence[i]

            prev_tbs = tb_groups[prev_dst]
            curr_tbs = tb_groups[curr_dst]

            # 检查 instance 数量是否匹配
            if len(prev_tbs) != len(curr_tbs):
                print(f"Warning: GPU {rank} has mismatched instance counts between dst {prev_dst} and {curr_dst}.")
                continue

            # 为当前 step 的每个 instance 添加对上一个 step 对应 instance 的依赖
            for j in range(len(curr_tbs)):
                curr_tb = curr_tbs[j]
                prev_tb = prev_tbs[j]
                prev_tb_id = prev_tb.get('id')

                # 修改 XML 中的 <step> 标签
                # 假设每个 send tb 里只有一个 step (s="0")
                step_elem = curr_tb.find('step')
                if step_elem is not None:
                    step_elem.set('depid', prev_tb_id)
                    step_elem.set('deps', '0')   # 依赖于目标 TB 的 step 0
                    step_elem.set('hasdep', '1') # 标记有依赖

    # 美化并保存
    indent(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Successfully processed. Output saved to '{output_file}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add serial dependencies to MSCCL XML based on linear shift steps.")
    parser.add_argument('--input', type=str, default='linear_shift.xml', help='Input XML filename')
    parser.add_argument('--output', type=str, default='linear_shift_dep.xml', help='Output XML filename')
    
    args = parser.parse_args()
    
    add_dependencies(args.input, args.output)