"""Visualization demo for temporal embeddings in Pokemon MD agent."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import time
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.extractor import QwenEmbeddingExtractor, EmbeddingMode
from src.embeddings.temporal_silo import TemporalSiloManager
from src.embeddings.vector_store import VectorStore


class EmbeddingVisualizer:
    """Visualize embeddings and temporal silo patterns."""
    
    def __init__(self):
        """Initialize visualizer."""
        self.extractor = QwenEmbeddingExtractor(model_name="Qwen3-VL-4B")
        self.silo_manager = TemporalSiloManager(base_fps=30)
        self.vector_store = VectorStore(backend="memory", embedding_dimension=1024)
    
    def simulate_trajectory(self, num_steps: int = 50) -> List[Dict[str, Any]]:
        """Simulate a Pokemon MD trajectory with embeddings.
        
        Args:
            num_steps: Number of steps in trajectory
            
        Returns:
            List of trajectory steps with embeddings
        """
        print(f"Simulating {num_steps}-step trajectory...")
        
        trajectory = []
        
        # Simulate different scenarios
        scenarios = [
            {"type": "exploration", "duration": 20},
            {"type": "combat", "duration": 15},
            {"type": "item_collection", "duration": 15},
        ]
        
        current_time = time.time()
        step_count = 0
        
        for scenario in scenarios:
            print(f"  Scenario: {scenario['type']} ({scenario['duration']} steps)")
            
            for i in range(scenario["duration"]):
                if step_count >= num_steps:
                    break
                
                # Simulate different embedding types for different scenarios
                if scenario["type"] == "combat":
                    embedding_mode = EmbeddingMode.THINK_INPUT
                elif scenario["type"] == "item_collection":
                    embedding_mode = EmbeddingMode.THINK_IMAGE_INPUT
                else:
                    embedding_mode = EmbeddingMode.THINK_FULL
                
                input_data = {
                    "screenshot": f"frame_{step_count}",
                    "action": f"action_{step_count}",
                    "scenario": scenario["type"]
                }
                
                # Generate dummy embedding
                embedding = self.extractor.extract(
                    input_data=input_data,
                    mode=embedding_mode
                )
                
                # Store in temporal silos
                self.silo_manager.store(
                    embedding=embedding,
                    trajectory_id="demo_trajectory",
                    metadata={
                        "step": step_count,
                        "action": f"action_{step_count}",
                        "scenario": scenario["type"],
                        "position": (np.random.randint(100, 300), np.random.randint(100, 200))
                    },
                    current_time=current_time
                )
                
                # Add to vector store
                self.vector_store.add_entry(
                    entry_id=f"step_{step_count}",
                    embedding=embedding,
                    metadata={
                        "step": step_count,
                        "action": f"action_{step_count}",
                        "scenario": scenario["type"]
                    },
                    silo_id="temporal_4frame"
                )
                
                trajectory.append({
                    "step": step_count,
                    "embedding": embedding,
                    "scenario": scenario["type"],
                    "timestamp": current_time
                })
                
                current_time += 0.5  # 500ms between steps
                step_count += 1
            
            if step_count >= num_steps:
                break
        
        print(f"Generated trajectory with {len(trajectory)} steps")
        return trajectory
    
    def visualize_embedding_space(self, trajectory: List[Dict[str, Any]]) -> None:
        """Visualize embeddings in 2D space using PCA.
        
        Args:
            trajectory: List of trajectory steps
        """
        print("\n=== Embedding Space Visualization ===")
        
        # Extract embeddings and metadata
        embeddings = np.array([step["embedding"] for step in trajectory])
        scenarios = [step["scenario"] for step in trajectory]
        
        # Apply PCA for 2D visualization
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        # PCA
        pca = PCA(n_components=2)
        embeddings_2d_pca = pca.fit_transform(embeddings)
        
        # t-SNE
        if len(embeddings) > 3:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            embeddings_2d_tsne = tsne.fit_transform(embeddings)
        else:
            embeddings_2d_tsne = embeddings_2d_pca
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Color mapping for scenarios
        scenario_colors = {
            "exploration": "blue",
            "combat": "red", 
            "item_collection": "green"
        }
        
        # PCA plot
        for scenario in set(scenarios):
            mask = [s == scenario for s in scenarios]
            ax1.scatter(
                embeddings_2d_pca[mask, 0],
                embeddings_2d_pca[mask, 1],
                c=scenario_colors[scenario],
                label=scenario,
                alpha=0.7,
                s=50
            )
        
        ax1.set_title("Embedding Space (PCA)")
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # t-SNE plot
        for scenario in set(scenarios):
            mask = [s == scenario for s in scenarios]
            ax2.scatter(
                embeddings_2d_tsne[mask, 0],
                embeddings_2d_tsne[mask, 1],
                c=scenario_colors[scenario],
                label=scenario,
                alpha=0.7,
                s=50
            )
        
        ax2.set_title("Embedding Space (t-SNE)")
        ax2.set_xlabel("t-SNE 1")
        ax2.set_ylabel("t-SNE 2")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("embedding_space_visualization.png", dpi=300, bbox_inches='tight')
        print("Saved embedding space visualization to 'embedding_space_visualization.png'")
        
        # Show PCA explained variance
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")
        
        plt.show()
    
    def visualize_temporal_silos(self) -> None:
        """Visualize temporal silo data distribution."""
        print("\n=== Temporal Silo Visualization ===")
        
        # Get silo statistics
        silo_stats = self.silo_manager.get_silo_stats()
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Silo utilization
        silo_ids = list(silo_stats.keys())
        utilizations = [silo_stats[silo_id]["utilization"] for silo_id in silo_ids]
        capacities = [silo_stats[silo_id]["max_capacity"] for silo_id in silo_ids]
        entries = [silo_stats[silo_id]["total_entries"] for silo_id in silo_ids]
        
        x_pos = range(len(silo_ids))
        
        ax1.bar(x_pos, entries, alpha=0.7, color='skyblue', label='Entries')
        ax1.bar(x_pos, [cap - ent for cap, ent in zip(capacities, entries)], 
                bottom=entries, alpha=0.3, color='gray', label='Available')
        
        ax1.set_title('Temporal Silo Utilization')
        ax1.set_xlabel('Silo ID')
        ax1.set_ylabel('Number of Entries')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([silo_id.replace('temporal_', '') for silo_id in silo_ids], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Time span vs configured span
        actual_spans = [silo_stats[silo_id]["actual_time_span"] for silo_id in silo_ids]
        configured_spans = [silo_stats[silo_id]["configured_time_span"] for silo_id in silo_ids]
        
        ax2.plot(x_pos, configured_spans, 'o-', label='Configured Time Span', linewidth=2, markersize=8)
        ax2.plot(x_pos, actual_spans, 's-', label='Actual Time Span', linewidth=2, markersize=8)
        
        ax2.set_title('Time Span: Configured vs Actual')
        ax2.set_xlabel('Silo')
        ax2.set_ylabel('Time Span (seconds)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([silo_id.replace('temporal_', '') for silo_id in silo_ids], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Sample rate visualization
        sample_rates = [silo_stats[silo_id]["sample_rate_ms"] for silo_id in silo_ids]
        
        ax3.bar(x_pos, sample_rates, alpha=0.7, color='orange')
        ax3.set_title('Sample Rates by Silo')
        ax3.set_xlabel('Silo')
        ax3.set_ylabel('Sample Rate (ms)')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([silo_id.replace('temporal_', '') for silo_id in silo_ids], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Silo hierarchy visualization
        hierarchy_info = {
            'temporal_1frame': 'Immediate (0-4s)',
            'temporal_2frame': 'Combat (0-8s)',
            'temporal_4frame': 'Navigation (0-16s)',
            'temporal_8frame': 'Room (0-32s)',
            'temporal_16frame': 'Floor (0-64s)',
            'temporal_32frame': 'Plan (0-128s)',
            'temporal_64frame': 'Cross-floor (2+min)'
        }
        
        # Create hierarchy diagram
        hierarchy_positions = {
            'temporal_1frame': (0, 6),
            'temporal_2frame': (1, 5),
            'temporal_4frame': (2, 4),
            'temporal_8frame': (3, 3),
            'temporal_16frame': (4, 2),
            'temporal_32frame': (5, 1),
            'temporal_64frame': (6, 0)
        }
        
        ax4.set_xlim(-0.5, 6.5)
        ax4.set_ylim(-0.5, 6.5)
        
        for silo_id, (x, y) in hierarchy_positions.items():
            if silo_id in silo_stats:
                entries = silo_stats[silo_id]["total_entries"]
                # Size based on number of entries
                size = max(100, entries * 50)
                
                ax4.scatter(x, y, s=size, alpha=0.6, c='lightblue', edgecolors='black')
                ax4.text(x, y-0.3, silo_id.replace('temporal_', ''), 
                        ha='center', va='top', fontsize=8, rotation=0)
                ax4.text(x, y-0.5, hierarchy_info[silo_id], 
                        ha='center', va='top', fontsize=7, style='italic')
        
        # Draw connections
        hierarchy_order = ['temporal_1frame', 'temporal_2frame', 'temporal_4frame', 
                          'temporal_8frame', 'temporal_16frame', 'temporal_32frame', 'temporal_64frame']
        
        for i in range(len(hierarchy_order) - 1):
            x1, y1 = hierarchy_positions[hierarchy_order[i]]
            x2, y2 = hierarchy_positions[hierarchy_order[i+1]]
            ax4.plot([x1, x2], [y1, y2], 'k--', alpha=0.3)
        
        ax4.set_title('Temporal Silo Hierarchy')
        ax4.set_xlabel('Temporal Resolution →')
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig("temporal_silos_visualization.png", dpi=300, bbox_inches='tight')
        print("Saved temporal silos visualization to 'temporal_silos_visualization.png'")
        plt.show()
    
    def visualize_search_results(self) -> None:
        """Visualize search results across silos."""
        print("\n=== Cross-Silo Search Visualization ===")
        
        # Generate a query embedding
        query_embedding = np.random.normal(0, 0.1, 1024)
        
        # Perform cross-silo search
        silo_results = self.silo_manager.cross_silo_search(
            query_embedding=query_embedding,
            top_k=5
        )
        
        # Visualize results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Results per silo
        silo_names = []
        similarities = []
        
        for silo_id, matches in silo_results.items():
            if matches:
                silo_names.append(silo_id.replace('temporal_', ''))
                avg_similarity = np.mean([sim for _, sim in matches])
                similarities.append(avg_similarity)
        
        if silo_names:
            bars = ax1.bar(silo_names, similarities, alpha=0.7, color='lightcoral')
            ax1.set_title('Average Similarity by Silo')
            ax1.set_xlabel('Silo')
            ax1.set_ylabel('Average Similarity')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, sim in zip(bars, similarities):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{sim:.3f}', ha='center', va='bottom')
        else:
            ax1.text(0.5, 0.5, 'No search results found', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('No Search Results')
        
        # 2. Similarity distribution
        all_similarities = []
        for matches in silo_results.values():
            for _, similarity in matches:
                all_similarities.append(similarity)
        
        if all_similarities:
            ax2.hist(all_similarities, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            ax2.axvline(np.mean(all_similarities), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_similarities):.3f}')
            ax2.set_title('Similarity Score Distribution')
            ax2.set_xlabel('Similarity Score')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No similarity scores available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('No Similarity Data')
        
        plt.tight_layout()
        plt.savefig("search_results_visualization.png", dpi=300, bbox_inches='tight')
        print("Saved search results visualization to 'search_results_visualization.png'")
        plt.show()
    
    def run_full_demo(self) -> None:
        """Run the complete embedding visualization demo."""
        print("Pokemon MD Agent - Embedding Visualization Demo")
        print("=" * 50)
        
        # 1. Generate trajectory
        trajectory = self.simulate_trajectory(num_steps=30)
        
        # 2. Visualize embedding space
        self.visualize_embedding_space(trajectory)
        
        # 3. Visualize temporal silos
        self.visualize_temporal_silos()
        
        # 4. Visualize search results
        self.visualize_search_results()
        
        # 5. Print summary statistics
        self.print_summary_statistics(trajectory)
        
        print("\n=== Demo Complete ===")
        print("Generated visualizations:")
        print("1. embedding_space_visualization.png - 2D embedding projections")
        print("2. temporal_silos_visualization.png - silo utilization and hierarchy")
        print("3. search_results_visualization.png - cross-silo search analysis")
    
    def print_summary_statistics(self, trajectory: List[Dict[str, Any]]) -> None:
        """Print summary statistics about the trajectory and embeddings.
        
        Args:
            trajectory: List of trajectory steps
        """
        print("\n=== Summary Statistics ===")
        
        # Trajectory statistics
        scenarios = [step["scenario"] for step in trajectory]
        scenario_counts = {}
        for scenario in scenarios:
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        
        print(f"Total trajectory steps: {len(trajectory)}")
        print("Scenario distribution:")
        for scenario, count in scenario_counts.items():
            percentage = (count / len(trajectory)) * 100
            print(f"  {scenario}: {count} steps ({percentage:.1f}%)")
        
        # Silo statistics
        silo_stats = self.silo_manager.get_silo_stats()
        print(f"\nTemporal silo utilization:")
        total_entries = 0
        for silo_id, stats in silo_stats.items():
            utilization = stats["utilization"]
            total_entries += stats["total_entries"]
            print(f"  {silo_id}: {stats['total_entries']}/{stats['max_capacity']} "
                  f"({utilization*100:.1f}% utilized)")
        
        print(f"Total entries across all silos: {total_entries}")
        
        # Vector store statistics
        store_stats = self.vector_store.get_stats()
        print(f"\nVector store: {store_stats['total_entries']} entries, "
              f"{store_stats['embedding_dimension']} dimensions")


def main():
    """Run the embedding visualization demo."""
    try:
        visualizer = EmbeddingVisualizer()
        visualizer.run_full_demo()
    except ImportError as e:
        print(f"Missing required dependencies: {e}")
        print("Install with: pip install matplotlib scikit-learn")
        return 1
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
