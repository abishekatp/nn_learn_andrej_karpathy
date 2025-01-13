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

impl Neuron {
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

    pub fn new_custom(neuron_inp: usize, activation_type: ActivationType) -> Self {
        let mut ner = Self::new(neuron_inp);
        ner.activation_type = activation_type;
        ner
    }

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
    pub fn new(neuron_inp: usize, layer_out: usize) -> Self {
        let mut neurons = vec![];
        for _ in 0..layer_out {
            neurons.push(Neuron::new(neuron_inp));
        }
        Self { neurons }
    }

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

    pub fn forward(&self, input: Vec<MVal>) -> Vec<MVal> {
        let mut outs = vec![];
        for n in &self.neurons {
            outs.push(n.forward(input.clone()));
        }
        outs
    }

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
    /// new MLP with Tanh activation.
    /// neuron_inp - number of inputs for each neuron.
    /// layer_outs - number of outputs(or neurons) in each layer.
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

    /// new MLP with custom activation like ReLU, Linear.
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

    // inp - length should be equal to no of inputs of each neuron.
    pub fn forward<T: IntoValue>(&self, inp: Vec<T>) -> Vec<MVal> {
        let mut input: Vec<_> = inp.into_iter().map(|v| MVal::new(v)).collect();
        for n in &self.layers {
            // for the next layer the input will be current layers output.
            input = n.forward(input.clone());
        }
        input
    }

    pub fn parameters(&mut self) -> Vec<MVal> {
        let mut params = vec![];
        for n in &self.layers {
            params.append(&mut n.parameters());
        }
        params
    }
}
