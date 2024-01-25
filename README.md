# IT3105 - Project 1
## Plants
* bathtub.py
* cournot.py 

The plant is any system whose behavior the controller will try to regulate.
Plants behaviour = plants output, can be viewed as a function that converts inputs to one or more outputs. This output
/behaviour is symbolized as

### Controllers
* ClassicPIDController.py
* GenericControllerModel.py
* NeuralPIDController.py

#### Controller output
The output of the controller:
* Is known as **control value** and is symbolised as `Y`. 
* Is one of the inputs for the plant function

#### Controller input
The input of the controller:
* Is an error term (`E`) 
  * represents the difference between goal/target (`T`) behaviour and its actual behaviour (`Y`)

### External dependencies
* JAX

