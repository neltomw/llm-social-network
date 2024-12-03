from constants_and_utils import *
from generate_personas import *

import os
import networkx as nx
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def parse_network_from_xml(xml_path):
    """
    Parse XML file and return a NetworkX graph.
    """
    # Parse XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Create empty directed graph
    G = nx.DiGraph()
    
    # Find all link elements (assuming structure matches sample)
    # Navigate through the XML hierarchy to find the network section
    networks = root.find('.//networks')
    if networks is None:
        print(f"No networks found in {xml_path}")
        return None
        
    # Get the first network (modify if you need to handle multiple networks)
    network = networks.find('network')
    if network is None:
        print(f"No network element found in {xml_path}")
        return None
    
    # Process all links
    for link in network.findall('link'):
        source = link.get('source')
        target = link.get('target')
        weight = float(link.get('value', 1.0))  # Default weight of 1.0 if not specified
        
        # Add edge to graph with weight
        G.add_edge(source, target, weight=weight)
    
    return G

def process_network_files(input_dir, output_dir):
    """
    Process all XML files in input directory and create network visualizations.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of XML files
    xml_files = list(Path(input_dir).glob('*.xml'))
    
    if not xml_files:
        print(f"No XML files found in {input_dir}")
        return
    
    print(f"Found {len(xml_files)} XML files to process")
    
    # Process each file
    for xml_path in xml_files:
        try:
            print(f"Processing {xml_path.name}...")
            
            # Parse network from XML
            G = parse_network_from_xml(xml_path)
            
            if G is None:
                continue
                
            # Generate output filename
            save_prefix = xml_path.stem  # filename without extension
            
            # Draw and save network plot
            draw_and_save_real_network_plot(G, save_prefix)
            
            print(f"Successfully processed {xml_path.name}")
            
        except Exception as e:
            print(f"Error processing {xml_path.name}: {str(e)}")

def main():
    # Define input and output directories
    input_dir = "real_networks"  # Directory containing XML files
    output_dir = plotting.PATH_TO_SAVED_PLOTS  # Using the path from your existing code
    
    # Process all network files
    process_network_files(input_dir, output_dir)

if __name__ == "__main__":
    main()