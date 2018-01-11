#pragma once

#include <list>
#include <vector>
#include <queue>
#include <boost/range/adaptor/reversed.hpp>

#include "Node.h"

namespace miniflow
{
	class Graph
	{
		/*
			Stores computational graph in topological order.
			Input nodes are calculated first.
		*/

		std::list<Node*> nodes_;

		// Traverse the graph from the top to the bottom
		template<typename F>
		void traverse(Node* node, F fn)
		{
			fn(node);
			for (Node* input : node->inbound_nodes_)
			{
				traverse(input, fn);
			}
		}

		void topological_sort(Node* output_node)
		{
			/*
				Sort the nodes in topological order
			*/

			std::list<Node*> input_nodes;

			traverse(output_node, [&](Node* node)
			{
				if (node->is_input()) input_nodes.push_front(node);
				else nodes_.push_front(node);
			});

			for (Node* node : input_nodes)
			{
				nodes_.push_front(node);
			}
		}

	public:

		Graph(Node& output_node)
		{
			topological_sort(&output_node);
		}

		// Returns the result of computation
		Tensor value() const
		{
			return nodes_.back()->getValue();
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
			for (Node* node : boost::adaptors::reverse(nodes_))
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

		void SGD(Scalar learning_rate, int repeats)
		{
			for (size_t i = 0; i <  repeats; i++)
			{
				Tensor cost = value();
				SGD_step(learning_rate);
			}
		}
	};
}