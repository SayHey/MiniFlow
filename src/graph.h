#pragma once
#include "node.h"

namespace miniflow
{
	class Graph
	{
		/*
			Stores computational graph in topological order.
			Input nodes are calculated first.
		*/

		std::list<NodeInterface*> nodes_;

		// Traverse the graph from the top to the bottom
		template<typename F>
		void traverse(NodeInterface* node, F fn)
		{
			fn(node);
			for (NodeInterface* input : node->inbound_nodes())
			{
				traverse(input, fn);
			}
		}

		void topological_sort(NodeInterface* output_node)
		{
			/*
				Sort the nodes in topological order
			*/

			std::list<NodeInterface*> input_nodes;

			traverse(output_node, [&](NodeInterface* node)
			{
				if (node->is_input()) input_nodes.push_front(node);
				else nodes_.push_front(node);
			});

			for (NodeInterface* node : input_nodes)
			{
				nodes_.push_front(node);
			}
		}

	public:

		explicit Graph(NodeInterface& output_node)
		{
			topological_sort(&output_node);
		}

		// Performs a forward pass through a list of Nodes.
		void forward()
		{
			for (NodeInterface* node : nodes_)
			{
				node->forward();
			}
		}

		// Performs a backward pass through a list of Nodes.
		void backward()
		{
			for (NodeInterface* node : boost::adaptors::reverse(nodes_))
			{
				node->backward();
			}
		}

		// Performs an update of all the trainable Nodes.
		void update(Scalar learning_rate)
		{
			for (NodeInterface* node : nodes_)
			{
				node->update(learning_rate);
			}
		}

		void SGD_step(Scalar learning_rate)
		{			
			forward();
			backward();
			update(learning_rate);			
		}

		void SGD(Scalar learning_rate, int repeats)
		{
			for (size_t i = 0; i < repeats; i++)
			{
 				SGD_step(learning_rate);
			}
		}
	};
}