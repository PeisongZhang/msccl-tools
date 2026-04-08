import xml.etree.ElementTree as ET
import argparse
import sys

def indent(elem, level=0):
    """Helper function to format XML output with indentation."""
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def add_ring_send_dependency(input_file, output_file, num_gpus, gpus_per_node):
    tree = ET.parse(input_file)
    root = tree.getroot()

    num_nodes = num_gpus // gpus_per_node

    if num_nodes <= 2:
        print(f"Info: num_nodes ({num_nodes}) <= 2. No serialization needed.")
        tree.write(output_file, encoding='utf-8', xml_declaration=False)
        return

    # 1. 扫描当前XML中已使用的最大depid
    max_depid = 0
    for step in root.findall(".//step"):
        depid = int(step.get('depid', -1))
        if depid > max_depid:
            max_depid = depid
    
    next_free_depid = max_depid + 1
    links_added = 0

    # 2. 遍历每个GPU
    for rank in range(num_gpus):
        gpu_node = root.find(f"./gpu[@id='{rank}']")
        if gpu_node is None:
            continue

        # 计算 Phase 2 Inter-node 的目标 Rank 序列
        n = rank // gpus_per_node
        g = rank % gpus_per_node
        
        inter_node_dsts = []
        # Step 0 是 Intra-node，从 Step 1 开始是 Inter-node
        for step in range(1, num_nodes):
            dst_n = (n + step) % num_nodes
            dst_rank = g + dst_n * gpus_per_node
            inter_node_dsts.append(dst_rank)

        # 3. 按 Channel 分组查找 TB
        chan_tb_map = {}
        for tb in gpu_node.findall("tb"):
            chan = int(tb.get("chan"))
            send_rank = int(tb.get("send"))
            
            if chan not in chan_tb_map:
                chan_tb_map[chan] = {}
            chan_tb_map[chan][send_rank] = tb

        # 4. 在每个 Channel 内建立发送链
        for chan, send_map in chan_tb_map.items():
            # 构建 TB 链: TB(Step 1) -> TB(Step 2) -> ...
            tb_chain = []
            chain_complete = True
            for dst in inter_node_dsts:
                if dst in send_map:
                    tb_chain.append(send_map[dst])
                else:
                    chain_complete = False
                    break
            
            if not chain_complete or len(tb_chain) < 2:
                continue

            # 5. 注入依赖：只针对 type="s" 的步骤
            for i in range(len(tb_chain) - 1):
                prev_tb = tb_chain[i]
                curr_tb = tb_chain[i+1]

                # --- 处理上游 (Producer) ---
                # 找到前一个 TB 中的 Send 步骤
                prev_send_step = None
                for step in prev_tb.findall("step"):
                    if step.get("type") == "s":
                        prev_send_step = step
                        break
                
                if prev_send_step is None:
                    continue # 异常情况，跳过

                # 确保上游步骤产生 depid
                signal_id = int(prev_send_step.get("depid", -1))
                if signal_id == -1:
                    signal_id = next_free_depid
                    next_free_depid += 1
                    prev_send_step.set("depid", str(signal_id))
                
                # --- 处理下游 (Consumer) ---
                # 找到当前 TB 中的 Send 步骤
                curr_send_step = None
                curr_steps = sorted(curr_tb.findall("step"), key=lambda x: int(x.get("s")))
                
                for step in curr_steps:
                    if step.get("type") == "s":
                        curr_send_step = step
                        break
                
                if curr_send_step is None:
                    continue

                # 检查当前 Send 步骤是否已有依赖
                existing_deps = int(curr_send_step.get("deps", -1))
                
                if existing_deps == -1:
                    # Case A: 没有现有依赖，直接添加
                    curr_send_step.set("deps", str(signal_id))
                    curr_send_step.set("hasdep", "1")
                    links_added += 1
                else:
                    # Case B: 已经有依赖了 (非常罕见，通常是 Phase 1 的依赖，但 Phase 2 的 Send 
                    # 一般只依赖本 TB 的 NOP 或者没有依赖)。
                    # MSCCL XML 一个步骤只能有一个 deps ID。
                    # 如果必须等待两个信号，我们需要插入一个 NOP 在 Send 之前来汇聚信号，
                    # 或者串行化 NOP。
                    
                    # 策略：在 Send 步骤之前紧贴着插入一个 NOP 步骤专门用于等待上游 Send
                    
                    # 1. 找到 Send 步骤的 s 序号
                    send_idx = int(curr_send_step.get("s"))
                    
                    # 2. 将 Send 及之后的所有步骤 s + 1
                    for step in curr_steps:
                        s_val = int(step.get("s"))
                        if s_val >= send_idx:
                            step.set("s", str(s_val + 1))
                    
                    # 3. 插入新的 NOP 步骤
                    wait_step = ET.Element("step")
                    wait_step.set("s", str(send_idx)) # 占据原来的位置
                    wait_step.set("type", "nop")
                    wait_step.set("srcbuf", "i") # Dummy
                    wait_step.set("srcoff", "-1")
                    wait_step.set("dstbuf", "o") # Dummy
                    wait_step.set("dstoff", "-1")
                    wait_step.set("cnt", "0")
                    wait_step.set("depid", "-1")
                    wait_step.set("deps", str(signal_id)) # 等待上游
                    wait_step.set("hasdep", "1")
                    
                    curr_tb.insert(0, wait_step) # 这里的 insert index 不重要，重要的是 s 属性
                    # 但为了 XML 美观，我们需要重新排序元素或者简单 append 后排序
                    # ElementTree insert 按列表位置。我们根据 s 找到正确位置插入比较复杂，
                    # 简单方法是 append 然后不做物理排序，因为 MSCCL Loader 通常看 s 属性。
                    # 为了保持 XML 可读性，我们再次排序子元素
                    
                    # 重新整理 children
                    children = list(curr_tb)
                    curr_tb.clear()
                    # 包含新 step
                    children.append(wait_step)
                    children.sort(key=lambda x: int(x.get("s")))
                    for child in children:
                        curr_tb.append(child)
                        
                    links_added += 1

    indent(root)
    tree.write(output_file, encoding='utf-8', xml_declaration=False)
    print(f"Successfully processed XML.")
    print(f"  Nodes: {num_nodes}")
    print(f"  Added {links_added} serialized dependency links (Send-to-Send).")
    print(f"  Output saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serialize Send steps in MSCCL Ring Phase 2.")
    parser.add_argument("xml_file", type=str, help="Input XML file path")
    parser.add_argument("num_gpus", type=int, help="Number of GPUs")
    parser.add_argument("gpus_per_node", type=int, help="GPUs per node")
    parser.add_argument("--output", type=str, default="p2p_ring_mod.xml", help="Output XML file path")

    args = parser.parse_args()

    add_ring_send_dependency(args.xml_file, args.output, args.num_gpus, args.gpus_per_node)