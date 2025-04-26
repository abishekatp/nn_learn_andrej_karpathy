use crate::value::{data_type::IntoValue, MVal};
use rand::Rng;

#[derive(Debug, Clone)]
pub enum ActivationType {
    Linear,
    Tanh,
    ReLU,
}

#[derive(Debug)]
pub struct Neuron {
    weights: Vec<MVal>,
    b: MVal,
    activation_type: ActivationType,
}

/*
What is Neuron?
    - Each Neuron will store list of weights that are of type Value(or equivalently MVal) and
      one bias Value.

    - The lenght of a input array(or list) should be equal to the length of the
      list of weights of a Neuron.

    - A Neuron will adjust each of its weights to port itself to compute the expected output.
      Each weight of a Neuron are adjusted based on the corresponding `grad` property of each weight of the Neuron.

    - The `grad` property of each of its weights will tell us how a small change to that weight has
      affected its output for the last passed input. Based on this we can adjust its weight
      such that it minimises the difference between computed output and actual expected output.

    - By default each neuron will use the Tanh activation. which means the sum of multiplication
      of input values and their corresponding weights will be passed to the tanh function before
      returning the final output. But Neuron can also use ReLU or Linear activation function.
      The activation function is used to introduce non-linearity in the model. Otherwise
      the model might become just a linear regression model. Which is just a linear equation
      of the form y = mx + b.

What is Layer?
    - the Layer is a list of Neurons. Usually we will pass the same input to each Neuron of
      the first layer. Our intution is that each Neuron will learn differnt property of
      the same input instance.
    - Other inner Layers are expected to captures complex connections among these outer layer
      Neurons.

What is MLP?
    - The MLP stands for multi layer perceptron. As we have discussed in the Layer section
      it will have multiple layers of Neurons.

    - The first layer will directly depend on its inputs.
      The i'th layer will depend on outputs of the (i-1)'th layer.

    - If i'th layer has 10 Neurons, then it will have 10 outputs. Then each Neuron in the (i+1)'th
      layer will have 10 inputs and 10 corresponding weights and one bias Value.

How to Use it?
    - First define a MLP with number of input for each neuron in the first layer and
      number of neurons in each layer. For example in the following code we are defining a
      MLP with 2 inputs, 16 neurons in first layer, 16 neurons in the second layer and 1 neuron in
      the last layer.
            `let mut model = MLP::new(2, vec![16, 16, 1]);`

    - Then call the forward method on MLP with each input instance. The model will make a
      prediction and return a list of outputs. The length of output list will be equal to number of
      neurons in the last layer of the MLP.
            `let pre = model.forward(v);`

    - Each of the predicted output will be of type MVal(Rc<RefCell<Value>>). So you can construct a
      loss of type MVal on top of these predicted outputs.
            `loss = loss + (1.0 - pre.clone() * out.as_f64());`

    - The above loss calculation is just a simple example. But the actual loss can be computed using any
      logic based on the use case. Now for the loss of type MVal you can calculate gradient for all its depedencies using
            `loss.zero_grad();
             loss.backward();`

    - Since the computation of loss depends on all the weights of the neural MLP either directly or indirectly,
      calling `loss.backward()` will compute the gradient each of these weights in the MLP.

    - These gradient values will tell us how a small change to these weights
    will affect the loss value. Weather increasing or decreasing the weights will increase or decrease the loss.

    - Based on these gradient we can update the `data` field of the weights. Then go to the first step
    and continue this process with next input instance.

    - Remember that there are diffent ways of choosing the training data, computing the loss and
    updating the weights. These decisions are made based on the specific use case.
*/

impl Neuron {
    /// Creates a neuron with `neuron_inp` number of weights.
    /// Here default activation type will be Tanh
    pub fn new(neuron_inp: usize) -> Self {
        let mut weights = vec![];
        let mut rng = rand::thread_rng();
        // Generate a random floating-point number between -1 and 1
        for _ in 0..neuron_inp {
            weights.push(MVal::new(rng.gen_range(-1.0..=1.0)));
        }
        Neuron {
            weights,
            b: MVal::new(rng.gen_range(-1.0..=1.0)),
            activation_type: ActivationType::Tanh,
        }
    }

    /// Same as new but with custom activation functions.
    pub fn new_custom(neuron_inp: usize, activation_type: ActivationType) -> Self {
        let mut ner = Self::new(neuron_inp);
        ner.activation_type = activation_type;
        ner
    }

    /// Multiplies each value of input instance with each weight of the neuron
    /// add the bias to the final sum and compute the activation value for it.
    /// ouput = activation((w_1 * x_1 + w_2 * x_2 + ... + w_k * x_k) + b)
    pub fn forward(&self, input: Vec<MVal>) -> MVal {
        let mut sum = self.b.clone();
        let mut i = 0;
        for w in &self.weights {
            let inp = input.get(i).unwrap_or(&MVal::new(0.0)).clone();
            sum = sum + (w.clone() * inp);
            i += 1;
        }

        match self.activation_type {
            ActivationType::Tanh => sum.tanh(),
            ActivationType::ReLU => sum.relu(),
            ActivationType::Linear => sum,
        }
    }

    /// returns list of weights and bias. these are the parameters of a Neuron.
    pub fn parameters(&self) -> Vec<MVal> {
        let mut params = self.weights.clone();
        params.push(self.b.clone());
        params
    }
}

#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    /// Creates a Layer with `layer_out` number of neurons with each neuron having
    /// `neuron_inp` number of inputs their corresponding weights.
    pub fn new(neuron_inp: usize, layer_out: usize) -> Self {
        let mut neurons = vec![];
        for _ in 0..layer_out {
            neurons.push(Neuron::new(neuron_inp));
        }
        Self { neurons }
    }

    /// Same as new but with custom activation function for Neurons
    pub fn new_custom(
        neuron_inp: usize,
        layer_out: usize,
        activation_type: ActivationType,
    ) -> Self {
        let mut neurons = vec![];
        for _ in 0..layer_out {
            neurons.push(Neuron::new_custom(neuron_inp, activation_type.clone()));
        }
        Self { neurons }
    }

    /// passes same input instance to each Neuron of the current Layer
    /// and returns a list of predicted outputs
    pub fn forward(&self, input: Vec<MVal>) -> Vec<MVal> {
        let mut outs = vec![];
        for n in &self.neurons {
            outs.push(n.forward(input.clone()));
        }
        outs
    }

    // returns a list of parameters of all Neurons in the current Layer.
    pub fn parameters(&self) -> Vec<MVal> {
        let mut params = vec![];
        for n in &self.neurons {
            params.append(&mut n.parameters());
        }
        params
    }
}

#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    /// New MLP with Tanh activation.
    /// neuron_inp - number of inputs for each neuron in the first layer.
    /// layer_outs - number of outputs(or neurons) in each layer in the order of first to last layer.
    /// Remember that number of Neurons on the i'th layer will be the
    /// length of input for each Neuron in (i+1)'th layer
    pub fn new(neuron_inp: usize, layer_outs: Vec<usize>) -> Self {
        let mut lays = vec![];
        let mut no_of_input = neuron_inp;
        for layer_out in layer_outs {
            lays.push(Layer::new(no_of_input, layer_out));
            // for each inner layer no of input will be equal to the no of output of previous layer.
            no_of_input = layer_out;
        }
        Self { layers: lays }
    }

    /// new MLP with custom activation like ReLU, Linear for all Neurons.
    pub fn new_custom(
        neuron_inp: usize,
        layer_outs: Vec<usize>,
        activation_type: ActivationType,
    ) -> Self {
        let mut lays = vec![];
        let mut no_of_input = neuron_inp;
        for layer_out in layer_outs {
            lays.push(Layer::new_custom(
                no_of_input,
                layer_out,
                activation_type.clone(),
            ));
            // for the next layer the input will be current layers output.
            no_of_input = layer_out;
        }
        Self { layers: lays }
    }

    /// inp - length should be equal to no of inputs of each neuron in the first layer.
    /// The `forward` method will pass same input instance to each Neuron of the first layer.
    /// Then ouputs predicted by i'th Layer will be passed as input for
    /// each Neuron in the (i+1)'th Layer.
    pub fn forward<T: IntoValue>(&self, inp: Vec<T>) -> Vec<MVal> {
        let mut input: Vec<_> = inp.into_iter().map(|v| MVal::new(v)).collect();
        for n in &self.layers {
            // for the next layer the input will be current layers output.
            input = n.forward(input.clone());
        }
        input
    }

    /// returns list of all the weights and biases of all Neurons in the MLP.
    pub fn parameters(&mut self) -> Vec<MVal> {
        let mut params = vec![];
        for n in &self.layers {
            params.append(&mut n.parameters());
        }
        params
    }
}
