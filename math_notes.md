# ML Math Notes & Reminders

---
### 1. Linear Algebra: General ML:
#### 1.1 General Formulation of Loss with Linear Algebra

Least Squares Loss:

Given the prediction, or activation, for some model and a __single__ training example $x$ (which is really $x_i$ from the entire set of examples $I$):

$$y = x^T\beta + e$$

For $J$ features and $I$ training examples, where:
- $x^T$ is our input vector of length $J$.
- $y$ is our rank-0 (a number) output or activatiom, given the task at hand.
- $\beta$ is our learned weight of length $J$.
- $x^T\beta$ denotes the dot product between $x$ and $\beta$
- $e$ denotes our residual vector of length $Dim(y)$

Then, the predicted value of \( y \) is \( x\beta \) and the error/residual for each data point is defined as:
\[ e = y - x\beta \]

The sum of squared residuals (SSR) for all data points is then defined as:
\[ SSR = e^T e = (y - x^T\beta)^T (y - x^T\beta) \]

Expanding this:
\[ SSR = y^T y - y^T x^T\beta - \beta^T x y + \beta^T x^T x\beta \]

<!-- From our cross terms:
\[ -2y^T x\beta \]

Then we have:
\[ SSR = y^T y - 2y^T x\beta + \beta^T x^T x\beta \] -->

Simplifying each vector (i.e. v) with the definition of inner product combined with definition of Euclidean norm:
\[ v^T v = v v^T = ||v||^2\]

And then, knowing that $y$ is a number:
\[ y^T = y \]

Then we have: 
\[ L = SSR = ||y||^2 - 2y x^T\beta + ||\beta||^2 ||x||^2 \]

We now have a convex loss surface for $x_{i,j} \text{ }\epsilon \text{ }I x J$

Then the cost function, $J(L)$, is:
\[J(\beta) = \frac{1}{I} \sum_{i=1}^{I} \left( ||y_i||^2 - 2y_i x_i^T\beta + ||\beta||^2 ||x_i||^2 \right)\]

Where $I$ is our number of training examples, and the operations over $J$ total features take place through dot product operations (i.e. $v^T v$)


---

<!-- \[ SSR = ||y||^2 - 2y^T x\beta + \beta^T x^T x\beta \] -->


#### (2) find stack overflow article for (1)
---
### 1.2 L1 vs. L2 Regularization: Some Intuition for Both
- For L1, the gradient is either 1 or -1 (abs. value func.), except for when the corresponding weight is 0 (gradient undefined).
- So, L1-regularization will move any weight towards 0 with the same step size when performing gradient descent, regardless the weight's value. 
    - __In other words,__ because the gradient of `|x|` is constant, we'll end up at zero over enough iterations for those weights without significant predictive power over our training epochs.

- In contrast, L2 gives __quadratic penalization__, so gradient is linearly decreasing towards 0 as the corresponding weight approaches 0. __As opposed to L1, the gradient for L2 is linear in $w_{j}$.__  
    - So: L2-regularization will also move any weight towards 0, but it will take smaller and smaller steps as a weight approaches 0.
    - From physics: think dampened pendulum (stole from stack overflow). We might not oscillate to the negative "amplitude" (weight) on this iteration, as with such a pendulum. 
        - Rather, when performing the gradient updating step (each iteration), we get smaller weights for the whole set of $w_j$'s, preventing overfitting.
        - Thinking about a polynomial regression (including when we pass the output into a sigmoid function in logistic regression), our model could prevent overfitting (__robust to outliers__) this way by diminishing the contribution of higher-order terms/weights (i.e. $x^6$ terms and corresponding weights) without eliminating their potential for modeling our data entirely. 

--- 
Now we return to the math for L1 regularization (multi-feature ($j's$), __single example__ ($i's$)). Before we had:
\[ SSR = ||y||^2 - y^T x^T\beta - \beta^T x y + ||\beta||^2 ||x||^2 \]

Now let's add our penalization term for L1 regularization (remember, just for each training example $i$):
\[SSR_{L1} = SSR + \lambda ||\beta||_1\]


Where the 1-norm of a vector is defined as the sum of the absolute values of that vector's elements:
\[
||\beta||_1 = \sum_{j=1}^{J} |\beta_j|
\]

Satisfying the definition of L1-regularization in the context of this loss term. 



Now let's compute our gradients. Taking the derivative of $SSR_{L1}$ to compute the gradient, we first have the following for the original $SSR$ (without the L1 term): 
\[
\nabla_{\beta} SSR = -2yx + 2\beta ||x||^2
\]

The result of which is of length $J$ given we take the gradient with respect to $\beta$ of length $J$, and we differentiate $\beta$ by the chain rule as we would in derivatives of functions composed of numbers, rather than vectors.

Now we move to differentiating the L1 term by $\beta$, which comes out to be the sign function as with $\frac{d(|x|)}{dx}$ being equal to this. Reminder: we can't differentiate straight-away, given at $x=0$ it is non-differentiable. So for the $j$th component (dimension) of our gradient, we get:
<!-- Insert plot of abs(x) and sign(x) to illustrate here. -->

\[
\nabla_{\beta_j} (\lambda ||\beta_j||_1) = \lambda \cdot \text{sign}(\beta_j)
\]

Or, to represent the full $J$-dimensional gradient without subscripts:
\[
\nabla_{\beta} (\lambda ||\beta||_1) = \lambda \cdot \text{sign}(\beta)
\]

Putting it all together, for a single training example $x$ and $\beta,x$ both of length $J$ features (still), we have:

\[\nabla_{\beta} SSR_{L1} = -2yx + 2\beta ||x||^2 + \lambda \cdot \text{sign}(\beta)\]

---
Reminder: this is a vectorized approach over our $j$'s. Let's make this notationally clear, again, over our particular $j$th (out of $J$) features for a particular $i$th training example:


\[ \nabla_{\beta_j} SSR_{L1,i} = -2y_i x_{i,j} + 2\beta_j x_{i,j}^2 + \lambda \cdot \text{sign}(\beta_j) \]


And now we sum over $I$ examples to get our total cost to get $J$ components (dimensions) of our gradient:
\[ \nabla_{\beta_j} J_{L1}(\beta) = \sum_{i=1}^{I} \left( -2y_i x_{i,j} + 2\beta_j x_{i,j}^2 \right) + \lambda \cdot \text{sign}(\beta_j) \]

where we seek to miniimize the cost function $J_{L1}(\beta)$.

--- 
A couple of things about $\nabla_{\beta_j} J_{L1}(\beta)$ and $J_{L1}(\beta)$:
- $\lambda \cdot \text{sign}(\beta)$ is inevitably a constant: it returns -1 if $\beta$ is negative and 1 if $\beta$ is positive.
    - __However,__ $\text{sign}(0) = 0$, so with this penalty term: unless loss from least squares terms are sufficiently minimized for $\beta_j$, then $\beta_j$ will be forced to 0 by the updates to $\beta$ in training. 
- Reminder, summming over all $i$ examples, we have the following update step in which the loss function has been descended:
\[ \nabla_{\beta_j} J_{L1}(\beta) = \sum_{i=1}^{I} \left( -2y_i x_{i,j} + 2\beta_j x_{i,j}^2 \right) + \lambda \cdot \text{sign}(\beta_j) \]
\[\beta_{j, upd} = \beta_{j}-\alpha \nabla_{\beta_j} J_{L1}(\beta).\] 
- If other $\beta_j$ are significant enough to not be zeroed, we have achieved some feature selection across $J$ features.
- We must be careful of the size of $\lambda$ to avoid eliminating resolving power of our algorithm to more complex relationships manifested in our data between the features -  underfitting.

#### L2 Regularization: Extension

Following a similar procedure for L2-regularization, we end up with:

\[ \nabla_{\beta_j} J_{L2}(\beta) = \sum_{i=1}^{I} \left( -2y_i x_{i,j} + 2\beta_j x_{i,j}^2 \right) + 2 \lambda \cdot \beta_j \]

Now we have a linear penalty in each component of the gradient $\beta_j$. So, $\beta_{j, upd}$ from above, given our new penalty term in $\nabla_{\beta_j} J_{L2}(\beta)$, will update $\beta_j$ in some negative, linear proportion to the current value of $\beta_j$. 

The key with our pivot to L2-regularization is understanding that it $-2 \lambda \cdot \beta_j$ in the update (gradient) step will be akin to dampened motion in physics, but only on one side of 0 on the number line:
- Positive values of $\beta_j$ are encouraged to become less positive and closer to 0 by the L2 term of the gradient in the update step, $-2 \lambda \cdot \beta_j$. 
    - But because this is linearly coupled to the current $\beta_j$: we will get asymptotically close to 0, but $\beta_j$ will not be floored to zero. Think small values of $\beta_j$ making that negative step smaller and smaller. 
    - Same for negative values getting less negative, but never 0 due to this penalty.

- The upshot of our conclusion for L2-regularization is that we enforce smaller $\beta_j$'s in order to prevent our model from overfitting the data.

- BUT: 
    - We won't have feature selection
    - We can have underfitting if we don't tune $\lambda$ appropriately.
    - We also are saving more floating point numbers (likely) in our $\beta_j$'s (think very large $J$ many features)

- Espescially with the latter, we may want to consider __Elastic Net regularization__ in which L1 and L2 are used together, and we get a mix of both benefits. 
    - It has been beneficial, in my experience, to train using only L2 penalization, to begin with, and use a conservative L1 $\lambda$ value (__obviously make coefficients of L1 and L2 penalites not both $\lambda$__).
