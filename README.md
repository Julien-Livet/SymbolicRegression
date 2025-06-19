# SymbolicRegression

Let give some symbols like ```x1``` and ```x2```.
The algorithm will consider first the linear combinations ```a*x1+b``` and ```c*x2+d```, then extra start symbolic expressions defined with ```extra_start_sym_expr```.
Set ```verbose``` to True to get some information during regression.
During process, we can check some expressions with ```checked_sym_expr```.
It is possible to specify the cost function to optimize with ```elementwise_loss```.
```auto_ops``` is used to select best operators among available operators, ```auto_ops_depth``` is a 2-tuple with depth for unary and binary operators.
Then the first iteration starts:
- We apply the unary operators defined with ```unary_operators```, for example ```e*log(a*x1+b)+f``` and ```g*log(c*x2+d)+h```.
- We apply then the binary operators defined with ```binary_operators```, for example ```i*(a*x1+b+c*x2+d)+j``` same as ```k*x1+l*x2+m```, etc. We can skip some combinations according to symmetric binary operators defined with ```symmetric_binary_operators```.

```discard_previous_expr``` will discard previous expressions at each level of iteration, avoiding combinatory explosion. To avoid too this explosion, ```maxloss```, ```maxexpr```, ```group_expr_size``` and ```maxtask``` are provided. Use ```process_sym_expr``` to process only target expressions.
We can define the depth of operators with ```operator_depth``` and the maximal complexity of searched expressions with ```maxcomplexity```.
We can apply some weights to operators for depth with ```op_weights```.
At each combination, we compute an optimal expression. If ```discrete_param_values``` is empty, we search optimal parameters with ```scipy.fit_curve``` (```maxfev``` is used), else we use brute force algorithm if possible (```brute_force_limit``` is used), else a smallest brute force problem, else a curve fit algorithm followed by genetic algorithm with ```deap```. ```discrete_param_values``` accept integer or float values for each parameter, str to define integer ranges like "(3, 5)" for ```range(3, 5)```, or "(3, 6, 2)" for ```range(3, 6, 2)```, or real ranges like "[3, 5, 1]" for ```np.linspace(3, 5, 1)```.
We process like that until the computed loss is less than ```epsloss```.
```eps``` is used to round numeric values and compare with other expressions for example.
We can subsitute some target expressions with ```subs_expr```. We can also avoid some expressions with ```avoided_expr```.
It is possible to call user callback during process with ```callback```.

To use multiprocess functionalities, functions must be pickable and ```freeze_support``` must be enabled.
It is possible to export results to CSV file with ```csv_filename```.

For more information, see existing tests.
```pytest -v```
If pytest-xdist is installed:
```pytest -v -n auto```
