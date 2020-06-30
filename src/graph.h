#pragma once
#include "node.h"
#include <ranges>
namespace miniflow
{
	class Graph
	{
		/*
			Stores computational graph in topological order.
			Input nodes are calculated first.
		*/

		const std::list<Node*> nodes_;

		// Traverse the graph from the top to the bottom
		template<typename F>
		constexpr void traverse(Node* node, F fn) const
		{
			fn(node);
			for (Node* input : node->inbound_nodes())
			{
				traverse(input, fn);
			}
		}

		std::list<Node*> topological_sort(Node* output_node) const
		{
			/*
				Sort the nodes in topological order
			*/

            std::list<Node*> nodes;
			std::list<Node*> input_nodes;

			traverse(output_node, [&](Node* node)
			{
				if (node->is_input()) input_nodes.push_front(node);
				else nodes.push_front(node);
			});

			for (Node* node : input_nodes)
			{
				nodes.push_front(node);
			}

            return nodes;
		}

	public:

		explicit Graph(Node& output_node):
            nodes_(topological_sort(&output_node))
		{
		}

		// Performs a forward pass through a list of Nodes.
		void forward()
		{
			for (Node* node : nodes_)
			{
				node->forward();
			}
		}

		// Performs a backward pass through a list of Nodes.
		void backward()
		{
			for (Node* node : nodes_ | std::views::reverse)
			{
				node->backward();
			}
		}

		// Performs an update of all the trainable Nodes.
		void update(Scalar learning_rate)
		{
			for (Node* node : nodes_)
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

		void SGD(Scalar learning_rate, uint repeats)
		{
			for (uint i = 0; i < repeats; i++)
			{
 				SGD_step(learning_rate);
			}
		}
	};
}