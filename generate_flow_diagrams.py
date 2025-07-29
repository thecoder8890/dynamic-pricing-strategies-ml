#!/usr/bin/env python3
"""
Generate flow diagrams for the dynamic pricing system documentation.
Creates visual representations of the pricing algorithms and data flows.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
import os

def create_system_architecture_diagram():
    """Create system architecture flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define colors
    data_color = '#E8F4FD'
    algo_color = '#FFF2CC'
    interface_color = '#E1D5E7'
    flow_color = '#D5E8D4'
    
    # Title
    ax.text(5, 9.5, 'Dynamic Pricing System Architecture', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Data Layer
    data_box = FancyBboxPatch((0.5, 6.5), 2.5, 2,
                              boxstyle="round,pad=0.1",
                              facecolor=data_color, edgecolor='black', linewidth=2)
    ax.add_patch(data_box)
    ax.text(1.75, 8, 'DATA LAYER', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.75, 7.5, '• Data Generator', fontsize=10, ha='center')
    ax.text(1.75, 7.2, '• CSV Storage', fontsize=10, ha='center')
    ax.text(1.75, 6.9, '• Validation', fontsize=10, ha='center')
    
    # Algorithm Layer
    algo_box = FancyBboxPatch((3.5, 6.5), 3, 2,
                              boxstyle="round,pad=0.1",
                              facecolor=algo_color, edgecolor='black', linewidth=2)
    ax.add_patch(algo_box)
    ax.text(5, 8, 'ALGORITHM LAYER', fontsize=12, fontweight='bold', ha='center')
    ax.text(5, 7.5, '• Rule-based Pricing', fontsize=10, ha='center')
    ax.text(5, 7.2, '• ML Models', fontsize=10, ha='center')
    ax.text(5, 6.9, '• Optimization', fontsize=10, ha='center')
    
    # Interface Layer
    interface_box = FancyBboxPatch((7, 6.5), 2.5, 2,
                                   boxstyle="round,pad=0.1",
                                   facecolor=interface_color, edgecolor='black', linewidth=2)
    ax.add_patch(interface_box)
    ax.text(8.25, 8, 'INTERFACE LAYER', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.25, 7.5, '• Jupyter Notebook', fontsize=10, ha='center')
    ax.text(8.25, 7.2, '• Interactive Widgets', fontsize=10, ha='center')
    ax.text(8.25, 6.9, '• Visualizations', fontsize=10, ha='center')
    
    # Data Flow
    flow_box = FancyBboxPatch((1, 4), 8, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor=flow_color, edgecolor='black', linewidth=2)
    ax.add_patch(flow_box)
    ax.text(5, 5, 'DATA FLOW PIPELINE', fontsize=12, fontweight='bold', ha='center')
    
    # Flow steps
    steps = ['Parameters', 'Generation', 'Features', 'Processing', 'Output']
    step_positions = [1.5, 3, 4.5, 6, 7.5]
    
    for i, (step, pos) in enumerate(zip(steps, step_positions)):
        ax.text(pos, 4.5, step, fontsize=10, ha='center', fontweight='bold')
        if i < len(steps) - 1:
            ax.arrow(pos + 0.3, 4.5, 0.9, 0, head_width=0.1, head_length=0.1, 
                    fc='black', ec='black')
    
    # Arrows between layers
    arrow1 = ConnectionPatch((3, 7.5), (3.5, 7.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc="red")
    ax.add_patch(arrow1)
    
    arrow2 = ConnectionPatch((6.5, 7.5), (7, 7.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, 
                            mutation_scale=20, fc="red")
    ax.add_patch(arrow2)
    
    # Business Context
    ax.text(5, 2.5, 'BUSINESS CONTEXT', fontsize=14, fontweight='bold', ha='center')
    ax.text(5, 2, '• E-commerce Marketplace Simulation', fontsize=11, ha='center')
    ax.text(5, 1.6, '• Dynamic Pricing Strategy Optimization', fontsize=11, ha='center')
    ax.text(5, 1.2, '• Revenue Maximization & Customer Segmentation', fontsize=11, ha='center')
    ax.text(5, 0.8, '• Educational ML Framework', fontsize=11, ha='center')
    
    plt.tight_layout()
    plt.savefig('docs/system_architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_rule_based_flow_diagram():
    """Create rule-based pricing decision flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Title
    ax.text(5, 15.5, 'Rule-Based Pricing Decision Flow', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Colors for different types of nodes
    input_color = '#E8F4FD'
    decision_color = '#FFF2CC'
    action_color = '#FFE6CC'
    output_color = '#D5E8D4'
    
    # Start node
    start_box = FancyBboxPatch((4, 14), 2, 0.8,
                               boxstyle="round,pad=0.1",
                               facecolor=input_color, edgecolor='black', linewidth=2)
    ax.add_patch(start_box)
    ax.text(5, 14.4, 'Input: Original Price', fontsize=10, fontweight='bold', ha='center')
    
    # Inventory check decision
    inv_decision = FancyBboxPatch((3.5, 12.5), 3, 1,
                                  boxstyle="round,pad=0.1",
                                  facecolor=decision_color, edgecolor='black', linewidth=2)
    ax.add_patch(inv_decision)
    ax.text(5, 13, 'Check Inventory Level', fontsize=10, fontweight='bold', ha='center')
    
    # Inventory actions
    low_inv = FancyBboxPatch((0.5, 10.5), 2, 0.8,
                             boxstyle="round,pad=0.1",
                             facecolor=action_color, edgecolor='black', linewidth=1)
    ax.add_patch(low_inv)
    ax.text(1.5, 10.9, 'Low Stock\n+25% markup', fontsize=9, ha='center')
    
    high_inv = FancyBboxPatch((7.5, 10.5), 2, 0.8,
                              boxstyle="round,pad=0.1",
                              facecolor=action_color, edgecolor='black', linewidth=1)
    ax.add_patch(high_inv)
    ax.text(8.5, 10.9, 'High Stock\n-5% discount', fontsize=9, ha='center')
    
    normal_inv = FancyBboxPatch((4, 10.5), 2, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=action_color, edgecolor='black', linewidth=1)
    ax.add_patch(normal_inv)
    ax.text(5, 10.9, 'Normal Stock\nNo adjustment', fontsize=9, ha='center')
    
    # Holiday check
    holiday_decision = FancyBboxPatch((3.5, 8.5), 3, 1,
                                      boxstyle="round,pad=0.1",
                                      facecolor=decision_color, edgecolor='black', linewidth=2)
    ax.add_patch(holiday_decision)
    ax.text(5, 9, 'Check Holiday Status', fontsize=10, fontweight='bold', ha='center')
    
    # Holiday action
    holiday_action = FancyBboxPatch((1, 6.5), 2.5, 0.8,
                                    boxstyle="round,pad=0.1",
                                    facecolor=action_color, edgecolor='black', linewidth=1)
    ax.add_patch(holiday_action)
    ax.text(2.25, 6.9, 'Holiday Season\n+20% premium', fontsize=9, ha='center')
    
    # Weekend check
    weekend_decision = FancyBboxPatch((3.5, 4.5), 3, 1,
                                      boxstyle="round,pad=0.1",
                                      facecolor=decision_color, edgecolor='black', linewidth=2)
    ax.add_patch(weekend_decision)
    ax.text(5, 5, 'Check Weekend Status', fontsize=10, fontweight='bold', ha='center')
    
    # Weekend action
    weekend_action = FancyBboxPatch((1, 2.5), 2.5, 0.8,
                                    boxstyle="round,pad=0.1",
                                    facecolor=action_color, edgecolor='black', linewidth=1)
    ax.add_patch(weekend_action)
    ax.text(2.25, 2.9, 'Weekend\n+10% premium', fontsize=9, ha='center')
    
    # Customer segment check
    customer_decision = FancyBboxPatch((6, 2.5), 3, 1,
                                       boxstyle="round,pad=0.1",
                                       facecolor=decision_color, edgecolor='black', linewidth=2)
    ax.add_patch(customer_decision)
    ax.text(7.5, 3, 'Check Customer\nSegment', fontsize=10, fontweight='bold', ha='center')
    
    # Customer action
    loyal_action = FancyBboxPatch((6.5, 0.5), 2, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor=action_color, edgecolor='black', linewidth=1)
    ax.add_patch(loyal_action)
    ax.text(7.5, 0.9, 'Loyal Customer\n-10% discount', fontsize=9, ha='center')
    
    # Final output
    output_box = FancyBboxPatch((3.5, 0.5), 3, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=output_color, edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, 0.9, 'Final Price + Adjustments', fontsize=10, fontweight='bold', ha='center')
    
    # Arrows - main flow
    arrows = [
        ((5, 14), (5, 13.5)),      # Start to inventory
        ((4, 12.5), (1.5, 11.3)),  # To low inventory
        ((5, 12.5), (5, 11.3)),    # To normal inventory  
        ((6, 12.5), (8.5, 11.3)),  # To high inventory
        ((1.5, 10.5), (4, 9.5)),   # Low inv to holiday
        ((5, 10.5), (5, 9.5)),     # Normal inv to holiday
        ((8.5, 10.5), (6, 9.5)),   # High inv to holiday
        ((4, 8.5), (2.25, 7.3)),   # To holiday action
        ((5, 8.5), (5, 5.5)),      # To weekend check
        ((2.25, 6.5), (4.5, 5.5)), # Holiday to weekend
        ((4, 4.5), (2.25, 3.3)),   # To weekend action
        ((6, 4.5), (7.5, 3.5)),    # To customer check
        ((2.25, 2.5), (4.5, 1.3)), # Weekend to final
        ((7.5, 2.5), (7.5, 1.3)),  # Customer to loyal action
        ((6.5, 1.3), (6.5, 1.3)),  # Customer to final
        ((7.5, 0.5), (6.5, 0.9)),  # Loyal to final
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    
    # Decision labels
    ax.text(2.5, 12, '< 10', fontsize=8, ha='center', style='italic')
    ax.text(5, 12, '10-80', fontsize=8, ha='center', style='italic')
    ax.text(7.5, 12, '> 80', fontsize=8, ha='center', style='italic')
    ax.text(3, 8, 'True', fontsize=8, ha='center', style='italic')
    ax.text(6.5, 8, 'False', fontsize=8, ha='center', style='italic')
    ax.text(3, 4, 'True', fontsize=8, ha='center', style='italic')
    ax.text(6.5, 4, 'False', fontsize=8, ha='center', style='italic')
    ax.text(8, 2, 'Loyal', fontsize=8, ha='center', style='italic')
    
    plt.tight_layout()
    plt.savefig('docs/rule_based_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_ml_training_flow_diagram():
    """Create ML model training flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Machine Learning Model Training Flow', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Colors
    data_color = '#E8F4FD'
    process_color = '#FFF2CC'
    decision_color = '#FFE6CC'
    output_color = '#D5E8D4'
    
    # Flow steps
    steps = [
        ('Historical Data', 10.5, data_color),
        ('Data Preprocessing', 9, process_color),
        ('Feature Engineering', 7.5, process_color),
        ('Train/Test Split', 6, process_color),
        ('Model Training', 4.5, process_color),
        ('Model Validation', 3, decision_color),
        ('Performance Check', 1.5, decision_color),
        ('Model Deployment', 0.5, output_color)
    ]
    
    # Draw main flow
    for i, (step, y_pos, color) in enumerate(steps):
        if i == 5:  # Performance check - diamond shape
            # Create diamond for decision
            diamond = mpatches.RegularPolygon((5, y_pos), 4, radius=0.8, 
                                            orientation=np.pi/4,
                                            facecolor=color, edgecolor='black')
            ax.add_patch(diamond)
        else:
            box = FancyBboxPatch((3.5, y_pos-0.3), 3, 0.6,
                               boxstyle="round,pad=0.1",
                               facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(box)
        
        ax.text(5, y_pos, step, fontsize=10, fontweight='bold', ha='center')
        
        # Add arrows between steps
        if i < len(steps) - 1:
            next_y = steps[i+1][1]
            ax.annotate('', xy=(5, next_y + 0.3), xytext=(5, y_pos - 0.3),
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Add feedback loop for poor performance
    hyperparameter_box = FancyBboxPatch((0.5, 3), 2.5, 0.6,
                                       boxstyle="round,pad=0.1",
                                       facecolor=process_color, edgecolor='black', linewidth=1)
    ax.add_patch(hyperparameter_box)
    ax.text(1.75, 3.3, 'Hyperparameter\nTuning', fontsize=9, fontweight='bold', ha='center')
    
    # Feedback arrows
    ax.annotate('', xy=(1.75, 4.8), xytext=(4.2, 1.5),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='red',
                             connectionstyle="arc3,rad=0.3"))
    ax.annotate('', xy=(3.5, 4.5), xytext=(3, 3.3),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='red'))
    
    # Labels for decision paths
    ax.text(6.5, 1.5, 'R² ≥ 0.7', fontsize=9, ha='center', style='italic', color='green')
    ax.text(3, 2, 'R² < 0.7', fontsize=9, ha='center', style='italic', color='red')
    
    # Add technical details boxes
    details_box1 = FancyBboxPatch((7, 8.5), 2.5, 2,
                                 boxstyle="round,pad=0.1",
                                 facecolor='lightgray', edgecolor='black', linewidth=1)
    ax.add_patch(details_box1)
    ax.text(8.25, 9.8, 'FEATURES', fontsize=9, fontweight='bold', ha='center')
    ax.text(8.25, 9.4, '• Original Price', fontsize=8, ha='center')
    ax.text(8.25, 9.1, '• Inventory Level', fontsize=8, ha='center')
    ax.text(8.25, 8.8, '• Holiday/Weekend', fontsize=8, ha='center')
    ax.text(8.25, 8.5, '• Competitor Price', fontsize=8, ha='center')
    
    details_box2 = FancyBboxPatch((7, 4), 2.5, 2,
                                 boxstyle="round,pad=0.1",
                                 facecolor='lightgray', edgecolor='black', linewidth=1)
    ax.add_patch(details_box2)
    ax.text(8.25, 5.3, 'MODEL SPECS', fontsize=9, fontweight='bold', ha='center')
    ax.text(8.25, 4.9, '• Linear Regression', fontsize=8, ha='center')
    ax.text(8.25, 4.6, '• 80/20 Split', fontsize=8, ha='center')
    ax.text(8.25, 4.3, '• R² Metric', fontsize=8, ha='center')
    ax.text(8.25, 4.0, '• RMSE Validation', fontsize=8, ha='center')
    
    plt.tight_layout()
    plt.savefig('docs/ml_training_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_price_optimization_flow():
    """Create price optimization process flow diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(7, 7.5, 'Price Optimization Process Flow', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Colors
    input_color = '#E8F4FD'
    process_color = '#FFF2CC'
    analysis_color = '#FFE6CC'
    decision_color = '#F8CECC'
    output_color = '#D5E8D4'
    
    # Process steps
    steps = [
        ('Current Product\nState', 1, 6, input_color),
        ('Generate Price\nCandidates', 3, 6, process_color),
        ('Predict Demand\nfor Each Price', 5, 6, analysis_color),
        ('Calculate\nRevenue', 7, 6, analysis_color),
        ('Apply Business\nConstraints', 9, 6, process_color),
        ('Select Optimal\nPrice', 11, 6, decision_color),
        ('Validate & Deploy\nNew Price', 13, 6, output_color)
    ]
    
    # Draw process boxes
    for step, x, y, color in steps:
        box = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, step, fontsize=9, fontweight='bold', ha='center')
    
    # Draw arrows between steps
    for i in range(len(steps) - 1):
        start_x = steps[i][1] + 0.6
        end_x = steps[i+1][1] - 0.6
        ax.annotate('', xy=(end_x, 6), xytext=(start_x, 6),
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Add detailed sub-processes
    # Price generation details
    price_details = FancyBboxPatch((2.2, 4), 1.6, 1.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor='lightblue', edgecolor='gray', linewidth=1)
    ax.add_patch(price_details)
    ax.text(3, 4.9, 'Price Range:', fontsize=8, fontweight='bold', ha='center')
    ax.text(3, 4.6, '80% - 120%', fontsize=8, ha='center')
    ax.text(3, 4.3, 'of original', fontsize=8, ha='center')
    ax.text(3, 4.0, '20 test points', fontsize=8, ha='center')
    
    # Demand prediction details
    demand_details = FancyBboxPatch((4.2, 4), 1.6, 1.5,
                                   boxstyle="round,pad=0.1",
                                   facecolor='lightyellow', edgecolor='gray', linewidth=1)
    ax.add_patch(demand_details)
    ax.text(5, 4.9, 'Elasticity Model:', fontsize=8, fontweight='bold', ha='center')
    ax.text(5, 4.6, 'demand = base ×', fontsize=8, ha='center')
    ax.text(5, 4.3, '(orig_price /', fontsize=8, ha='center')
    ax.text(5, 4.0, 'test_price)^e', fontsize=8, ha='center')
    
    # Revenue calculation details
    revenue_details = FancyBboxPatch((6.2, 4), 1.6, 1.5,
                                    boxstyle="round,pad=0.1",
                                    facecolor='lightcoral', edgecolor='gray', linewidth=1)
    ax.add_patch(revenue_details)
    ax.text(7, 4.9, 'Revenue Calc:', fontsize=8, fontweight='bold', ha='center')
    ax.text(7, 4.6, 'Revenue =', fontsize=8, ha='center')
    ax.text(7, 4.3, 'Price × Demand', fontsize=8, ha='center')
    ax.text(7, 4.0, 'Find Maximum', fontsize=8, ha='center')
    
    # Constraints details
    constraints_details = FancyBboxPatch((8.2, 4), 1.6, 1.5,
                                        boxstyle="round,pad=0.1",
                                        facecolor='lightgray', edgecolor='gray', linewidth=1)
    ax.add_patch(constraints_details)
    ax.text(9, 4.9, 'Constraints:', fontsize=8, fontweight='bold', ha='center')
    ax.text(9, 4.6, '• Min margin', fontsize=8, ha='center')
    ax.text(9, 4.3, '• Max increase', fontsize=8, ha='center')
    ax.text(9, 4.0, '• Competition', fontsize=8, ha='center')
    
    # Connect detail boxes to main flow
    detail_connections = [(3, 5.6), (5, 5.6), (7, 5.6), (9, 5.6)]
    for x, y in detail_connections:
        ax.annotate('', xy=(x, y - 0.4), xytext=(x, 4 + 0.75),
                   arrowprops=dict(arrowstyle='->', lw=1, color='gray', alpha=0.7))
    
    # Add feedback loop
    feedback_box = FancyBboxPatch((10.5, 2), 2, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor='pink', edgecolor='red', linewidth=1)
    ax.add_patch(feedback_box)
    ax.text(11.5, 2.5, 'A/B Testing &', fontsize=9, fontweight='bold', ha='center')
    ax.text(11.5, 2.2, 'Performance', fontsize=9, fontweight='bold', ha='center')
    ax.text(11.5, 1.9, 'Monitoring', fontsize=9, fontweight='bold', ha='center')
    
    # Feedback arrow
    ax.annotate('', xy=(1.5, 5.5), xytext=(11.5, 3),
               arrowprops=dict(arrowstyle='->', lw=1.5, color='red',
                             connectionstyle="arc3,rad=0.4"))
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=input_color, label='Input Data'),
        mpatches.Patch(color=process_color, label='Processing'),
        mpatches.Patch(color=analysis_color, label='Analysis'),
        mpatches.Patch(color=decision_color, label='Decision'),
        mpatches.Patch(color=output_color, label='Output')
    ]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0, 0))
    
    plt.tight_layout()
    plt.savefig('docs/price_optimization_flow.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all flow diagrams"""
    # Create docs directory if it doesn't exist
    os.makedirs('docs', exist_ok=True)
    
    print("Generating system architecture diagram...")
    create_system_architecture_diagram()
    
    print("Generating rule-based pricing flow diagram...")
    create_rule_based_flow_diagram()
    
    print("Generating ML training flow diagram...")
    create_ml_training_flow_diagram()
    
    print("Generating price optimization flow diagram...")
    create_price_optimization_flow()
    
    print("All flow diagrams generated successfully!")
    print("Diagrams saved in 'docs/' directory:")
    print("  - system_architecture_diagram.png")
    print("  - rule_based_flow_diagram.png")
    print("  - ml_training_flow_diagram.png")
    print("  - price_optimization_flow.png")

if __name__ == "__main__":
    main()