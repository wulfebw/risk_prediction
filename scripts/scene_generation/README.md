<!-- MarkdownTOC -->

- [Fitting a Bayesian Network Scene Generation Model](#fitting-a-bayesian-network-scene-generation-model)
    - [Dataset options](#dataset-options)
    - [Bayes Net Description](#bayes-net-description)
    - [Bayes Net Automotive Scene Generation](#bayes-net-automotive-scene-generation)
    - [Fitting The Network](#fitting-the-network)
- [Fitting a Proposal Bayesian Network Scene Generation Model](#fitting-a-proposal-bayesian-network-scene-generation-model)
    - [Cross Entropy Method \(CEM\)](#cross-entropy-method-cem)
    - [CEM Applied to Fitting a Proposal BN](#cem-applied-to-fitting-a-proposal-bn)
    - [Visualizing Scenes and Aggregate Measures](#visualizing-scenes-and-aggregate-measures)

<!-- /MarkdownTOC -->

# Fitting a Bayesian Network Scene Generation Model

## Dataset options
- fitting a bayes net model first requires a dataset
- there are two options
1. fit a BN to NGSIM data
    - see docs/validation for how to extract a dataset from NGSIM
2. fit a BN to heuristically generated data
    - see docs/collection for how to collect a heuristically generated dataset

## Bayes Net Description
- A bayesian network is a probabilistic graphical model 
- It consists of a directed acyclic graph (DAG) between a set of random variables, each of which is associated with a conditional probability distribution
- The probability distribution for each variable depends only on the values of its parent variables
- A sample from a BN is an assignment to all of the variables in the BN
- Provided that you only condition root variables in the DAG, BNs have the nice property that sampling from them can be done quickly

## Bayes Net Automotive Scene Generation 
- For in-depth discussion of automotive scene generation see the following papers
    + http://timallanwheeler.com/aboutme/papers/wheeler2016factorgraph.pdf
    + http://timallanwheeler.com/aboutme/papers/wheeler2015initial.pdf
- Briefly, the scene generation model we use here is 'factored' across lanes in the sense that no correlations across lanes are captured
- Generation proceeds as follows
    + The variables defining the first vehicle in the scene are sampled
        * these are currently aggressiveness, attentiveness, vehicle length, vehicle width
            - variables that could be sampled, but in this first-vehicle case are marginalized are fore distance, fore velocity, and relative velocity
    + The state variables of the second vehicle are then sampled conditionally on the variable assignment for the first vehicle
    + The process repeats until the scene is completely generated
- This process makes the assumption that the state variables of one vehicle are conditionally independent of the state variables of vehicles more than one car in front given the variable assignment to the vehicle in front
- What this means is that 'fitting a bayes net scene generation model' just means 'fitting a BN over the state variables'
    + This still leaves a few questions, for example, how is the distribution over each variable defined?
    + We take the simple approach that each variable is discretized in bins for learning, and then for sampling each variable is assigned a category, which defines two edges of range, from which the continuous value (if applicable) is sampled uniformly

## Fitting The Network
- The two files relevant to BN fitting are `scene_generation/run_fit_bayes_net.jl` and `scene_generation/fit_bayes_net.jl`
- To fit a bayes net, edit `scene_generation/run_fit_bayes_net.jl`:
    + replace the filepath to the dataset 
    + replace the output filepath of where to save the bn 
    + run `julia run_fit_bayes_net.jl`
- This may take a while if the dataset is large
- A summary of what this does is
    1. it loads the dataset
    2. it performs some basic sanitation, for example throwing out samples that do not fall within a specified range
    3. it defines the DAG of the BN 
    4. it fits the BN to the dataset
- You may want to use different values than the defaults, in which case take a look at the functions themselves for description of variables

# Fitting a Proposal Bayesian Network Scene Generation Model
- A strength of scene generation models is that they define the probability distribution for every scene you generate
- One ability this confers is to generate scenes that are unlikely under the model, but that are desirable for some other reason
- An example is generating dangerous scenes because we want to focus on them
- If you had some way of generating only dangerous scenes with the model, then you would still know the probability of generating them, which would allow you to derive principled estimates of the e.g., likelihood of those events under the model
- The question is how to generate dangerous scenes?
    + I think there are at least two approaches to this
        1. Keep the same model, change the distributions over variables
        2. Add variables to the model reflecting the danger of the situation
    + It's not clear which is to be preferred, but we take the first approach
    + In particular, we use the cross entropy method to fit an alternative Bayes Net, which we refer to as the 'proposal BN'
    + Then, to generate a scene, we use the normal BN to generate the vehicles in the scene, except for one vehicle, which we generate using the proposal BN
        * You can then define the likelihood of having generated this proposal vehicle as the ratio of the likelihoods of the proposal and normal BN 
        * i.e., normal prob / proposal prob
        * This is the likelihood weight of the sample

## Cross Entropy Method (CEM)
- see the notebook `simple_cross_entropy_method_example.ipynb` for an explanation of the cross entropy method

## CEM Applied to Fitting a Proposal BN
- Fitting a proposal BN requires a base BN, so first fit a BN from data using the approach described above, and then fit the proposal BN
- Given the base BN, fit a proposal BN by running `run_fit_proposal_bayes_net.jl`
- Currently, this file loads settings from a dataset, this dataset should be the one used to fit the normal BN
    + This is not strictly necessary, but it is convenient because fitting a proposal BN via CEM requires running simulations, and the settings for this simulation must be defined
- Next, certain parameters of CEM are set, and the proposal network is generated
- See the file for details on the parameters, but one important thing to note is that this process can be run in parallel, and should be if possible since it takes a while to run
- To run a julia script with multiple processes add the `-p` flag
    + for example `julia -p 10 myscript.jl`, where 10 is the number of cores available

## Visualizing Scenes and Aggregate Measures
- Note that a scene generation model just defines the initial conditions of a scene, and not how that scenes evolves over time
- Because of this, we can fully evaluate a scene generation model based on the static scenes it generates
- One way to do this is qualitatively, by generating scenes an looking at them 
- Another way is to generate a large number of scenes, compute aggregate measures that we care about, and compare those measures to the values of the original dataset
- These tasks can be performed in the notebook `test_bayes_net_lane_generator.ipynb`
