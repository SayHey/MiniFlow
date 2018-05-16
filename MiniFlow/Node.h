#pragma once

#include "Common.h"
#include "Tensor.h"

namespace miniflow
{
	//using placeholder::Tensor;

	class NodeInterface
	{

	public:

		virtual ~NodeInterface() = default;
		virtual void forward() = 0;									// Calculates the node's output (value_)
		virtual void backward() = 0;								// Calculates derivatives (gradient_)
		virtual void update(Scalar /*learning_rate*/) = 0;				// Updates trainables
		virtual bool is_input() const = 0;							//
		virtual std::vector<NodeInterface*> inbound_nodes() = 0;	//
	};

	template<typename Tensor>
	class Node : public NodeInterface
	{
		/*
			Base class for nodes in the network.
		*/

	protected:

		struct OutboundNode
		{
			const Node* node;							//: A pointer to outbound node itself.
			std::size_t index;							//: An index of the host node in the outbound node's list of the inputs.

			Tensor getGradient()
			{
				return node->getGradient()[index];		// An index is used for getting the outbound node gradient with respect to host node.
			}
		};

		Tensor value_;									//: The eventual value of this node. Set by running the forward() method.
		std::vector<Node*> inbound_nodes_;				//: A list of nodes with edges into this node.
		std::vector<OutboundNode> outbound_nodes_;		//: A list of nodes that this node outputs to.		
		std::vector<Tensor> gradient_;					//: Partial derivatives of this node with respect to the input nodes.
														//  Set by running the forward() method.
														//  Has the same size as a list of the input nodes.
	public:

		Node(std::vector<Node*> inbound) :
			inbound_nodes_(inbound)
		{
			// Initialize value to 0.
			value_ = 0;
			// Initialize a partial derivative for each of the inbound_nodes to 0.
			gradient_.resize(inbound.size(), 0);
			// Sets this node as an outbound node for all of this node's inputs.
			for (std::size_t i = 0; i < inbound_nodes_.size(); i++)
			{
				inbound_nodes_[i]->outbound_nodes_.push_back({ this, 0 });
			}
		}

		// Node Interface virtual functions. General implementations.
		// Note that different functions are further overridden in Node specializations.
		void forward() override {};
		void backward() override {};
		void update(Scalar /*learning_rate*/) override {};
		bool is_input() const override { return false; }			
		std::vector<NodeInterface*> inbound_nodes() override
		{
			std::vector<NodeInterface*> inbound_nodes_interface(inbound_nodes_.size());
			for (std::size_t i = 0; i < inbound_nodes_.size(); i++)
			{
				inbound_nodes_interface[i] = inbound_nodes_[i];
			}
			return inbound_nodes_interface;
		}

		// Access functions.
		Tensor getValue() const { return value_; }
		std::vector<Tensor> getGradient() const { return gradient_; }
	};

	template<typename Tensor>
	class Input : public Node<Tensor>
	{
		/*
			A generic input into the network.
			Has no input nodes, but has a partial derivative of the cost with respect to this itself.
			gradient_ size is 1 and a partial derivative is stored in gradient_[0].
		*/

	public:

		Input(Tensor const& input) :
			Node(std::vector<Node*>(0))
		{
			gradient_.resize(1, 0);
			value_ = input;
		}

		void backward() override
		{
			// Cycle through the outputs. Sum the partial with respect to the input over all the outputs.
			for (OutboundNode outbound_node : outbound_nodes_)
			{
				// Get the partial of the cost with respect to this node.
				Tensor grad_cost = outbound_node.getGradient();
				gradient_[0] += outbound_node.getGradient();
			}
		}

		bool is_input() const override { return true; }
	};

	template<typename Tensor>
	class Trainable : public Input<Tensor>
	{
		/*
			A trainable parameter of the network.
		*/

	public:

		Trainable(Tensor const& input) :
			Input(input)
		{
		}

		// Performs SGD step
		void update(Scalar learning_rate) override
		{
			value_ -= learning_rate * gradient_[0];
		}
	};

	template<typename Tensor>
	class Linear : public Node<Tensor>
	{
		/*
			Represents a node that performs a linear transform.

			Input is {X, W, b}.
			Output is dot(X, W) + b.
		*/

	public:

		Linear(Node& X, Node& W, Node& b) :
			Node(std::vector<Node*>{ &X, &W, &b })
		{
		}

		void forward() override
		{
			// The math behind a linear transform.
			Tensor X = inbound_nodes_[0]->getValue();
			Tensor W = inbound_nodes_[1]->getValue();
			Tensor b = inbound_nodes_[2]->getValue();
			value_ = Tensor::dot(X, W) + b;
		}

		void backward() override
		{
			// Cycle through the outputs. Sum the partial with respect to the input over all the outputs.
			for (OutboundNode outbound_node : outbound_nodes_)
			{
				// Get the partial of the cost with respect to this node.
				Tensor grad_cost = outbound_node.getGradient();
				// Set the partial of the loss with respect to this node's inputs.
				gradient_[0] += Tensor::dot(grad_cost, inbound_nodes_[1]->getValue().T());
				// Set the partial of the loss with respect to this node's weights.
				gradient_[1] += Tensor::dot(inbound_nodes_[0]->getValue().T(), grad_cost);
				// Set the partial of the loss with respect to this node's bias.
				gradient_[2] += Tensor::sum(grad_cost); //TODO
			}
		}
	};

	template<typename Tensor>
	class Sigmoid : public Node<Tensor>
	{
		/*
			 Represents a node that performs the sigmoid activation function.

			 Input is {X}.
			 Output is sigmoid(X) = 1 / (1 + exp(-X));
		*/

	public:

		Sigmoid(Node& input) :
			Node(std::vector<Node*>{ &input })
		{
		}

		void forward() override
		{
			// The math behind a sigmoid.
			const Tensor& input = inbound_nodes_[0]->getValue();
			value_ = 1. / (1 + Tensor::exp(-input));
		}

		void backward() override
		{
			/*
				The derivative of sigmoid(X):
				d/dx[sigmoid(x)] = sigmoid(X) * (1 - sigmoid(X))
			*/

			// Cycle through the outputs. Sum the partial with respect to the input over all the outputs.
			for (OutboundNode outbound_node : outbound_nodes_)
			{
				// Get the partial of the cost with respect to this node.
				Tensor sigmoid = value_;
				Tensor grad_cost = outbound_node.getGradient();
				gradient_[0] += sigmoid * (1 - sigmoid) * grad_cost;
			}
		};
	};

	template<typename Tensor>
	class MSE : public Node<Tensor>
	{
		/*
		Represents a node that calculates mean squared error cost function.
		Should be used as the last node for a network.

		Input is {labels, predictions}.
		Output is mean squared error;
		*/

		//cached during forward() computation for backward
		std::size_t m_;
		Tensor diff_; 
		//

	public:

		MSE(Node& labels, Node& predictions) :
			Node(std::vector<Node*>{ &labels, &predictions })
		{
		}

		void forward() override
		{
			Tensor labels = inbound_nodes_[0]->getValue();
			Tensor predictions = inbound_nodes_[1]->getValue();

			m_ = 1;// inbound_nodes[0]->getValue().shape[0]; //TODO
			diff_ = labels - predictions;
			value_ = Tensor::mean(Tensor::sqr(diff_));
		}

		void backward() override
		{
			/*
				Calculates the gradient of the cost.
			*/		
			
			gradient_[0] = 2. / m_ * diff_;
			gradient_[1] = - gradient_[0];
		}
	};
}