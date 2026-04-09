# Deep learning optimization algorithm



## Deep learning optimization algorithms: SGD, RMSProp, AdaGrad, Adam detailed explanation

In the training process of deep learning, optimization algorithms play a crucial role. They determine how the model parameters are updated according to the gradient of the loss function, thus affecting the model's convergence speed and final performance. This article will introduce in depth several of the most commonly used and basic optimization algorithms: SGD, RMSProp, AdaGrad and Adam, and analyze their formulas, advantages and disadvantages.

### 1. Stochastic Gradient Descent (SGD)

SGD is a variant of the most basic gradient descent algorithm. Unlike batch gradient descent (Batch Gradient Descent), which calculates the gradient of all samples at once, SGD only uses one randomly selected sample to calculate the gradient and update parameters at a time.

#### formula

Assume that the loss function is $J(\theta)$ and the learning rate is $\eta$.
For parameter $\theta$:

$ \theta_{t+1} = \theta_t - \eta \nabla J(\theta_t; x^{(i)}; y^{(i)}) $

in:
*   $\theta_t$ is a parameter at time step $t$.
*   $\eta$ is the learning rate.
*   $\nabla J(\theta_t; x^{(i)}; y^{(i)})$ is the gradient calculated on sample $(x^{(i)}, y^{(i)})$.

#### advantage

*   **High computational efficiency:** Only one sample is used for each update, the calculation amount is small and the speed is fast.
*   **Ability to jump out of local optima:** Due to the stochastic nature of gradients, SGD has the potential to jump out of local minima and find better global minima.
*   **Low memory footprint:** No need to store gradients for the entire dataset.

#### shortcoming

*   **Oscillation in the convergence process:** Since only one sample is used at a time, the gradient direction may be unstable, resulting in large oscillations in the loss function during the convergence process.
*   **Difficulty in selecting the learning rate:** The learning rate needs to be carefully adjusted $\eta$. If it is too large, it may cause the oscillation to fail to converge, and if it is too small, the convergence will be slow.
*   **Not friendly to sparse data:** For sparse features, their gradients may be very small, resulting in slow updates.

### 2. AdaGrad (Adaptive Gradient)

The core idea of ​​the AdaGrad algorithm is to maintain a cumulative squared gradient for each parameter and adjust the learning rate of the parameter based on this cumulative value. It can automatically adjust the learning rate based on the historical gradient size of parameters, and performs better on sparse gradients.

#### formula

For parameter $\theta$:

1.  **Calculate the square of the current gradient:** $g_t^2$
2.  **Cumulative squared gradient:**
    $ R_t = R_{t-1} + g_t^2 $
Among them $R_0 = 0$.
3.  **Update parameters:**
    $ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{R_t + \epsilon}} \odot g_t $

in:
*   $\theta_t$ is a parameter at time step $t$.
*   $\eta$ is the global learning rate.
*   $g_t$ is the gradient calculated at time step $t$.
*   $R_t$ is the sum of all squared historical gradients of parameter $\theta$.
*   $\epsilon$ is a small constant (e.g. $10^{-8}$) that prevents division by zero.
*   $\odot$ means element-wise multiplication.

#### advantage

*   **Adaptive learning rate:** Automatically adjusts the learning rate for each parameter, providing a larger learning rate for sparse gradients (parameters that appear less frequently) and a smaller learning rate for frequently occurring parameters.
*   **No need to manually adjust the learning rate:** Reduces the difficulty of learning rate tuning to a certain extent.

#### shortcoming

*   **The learning rate decays too fast:** As the training progresses, $R_t$ will continue to accumulate and increase, causing the learning rate to continue to decrease, and the model may stop learning before it converges.
*   **Unable to handle non-stationary objectives:** AdaGrad's learning rate may decay too quickly for objective functions with large gradient changes.

### 3. RMSProp (Root Mean Square Propagation)

The RMSProp algorithm is an improvement on AdaGrad, which introduces the exponential decay average to calculate the accumulation of squared gradients instead of direct accumulation. This makes the learning rate decay more gently and avoids the problem of premature decay of AdaGrad's learning rate.

#### formula

For parameter $\theta$:

1.  **Calculate the square of the current gradient:** $g_t^2$
2.  **Compute the exponentially decaying average of squared gradients:**
    $ E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) g_t^2 $
Among them $E[g^2]_0 = 0$.
3.  **Update parameters:**
    $ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \odot g_t $

in:
*   $\theta_t$ is a parameter at time step $t$.
*   $\eta$ is the global learning rate.
*   $g_t$ is the gradient calculated at time step $t$.
*   $E[g^2]_t$ is the exponentially decaying average of the squared gradient of parameter $\theta$.
*   $\beta$ is the decay rate (usually set to 0.9).
*   $\epsilon$ is a small constant (e.g. $10^{-8}$).
*   $\odot$ means element-wise multiplication.

#### advantage

*   **Adaptive learning rate:** Inherits the advantages of AdaGrad and is able to adjust the learning rate for each parameter.
*   **Solve the problem of too fast decay of learning rate:** Avoid premature decay of learning rate through exponential decay average.
*   **Performs better on non-stationary targets:** More robust to situations with large gradient changes.

#### shortcoming

*   **Learning rate still needs to be adjusted:** Although the problem with AdaGrad is alleviated, the global learning rate $\eta$ still needs to be adjusted.
*   **No Momentum:** No momentum information from the gradient is taken into account.

### 4. Adam (Adaptive Moment Estimation)

The Adam algorithm combines the ideas of Momentum and RMSProp, while maintaining the first-order moment estimate (momentum) and second-order moment estimate (RMSProp) of the gradient, and performs bias correction.

#### formula

For parameter $\theta$:

1.  **Compute first moment estimate (momentum):**
    $ m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t $
2.  **Compute second moment estimate (RMSProp):**
    $ v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 $
3.  **Bias Correction:**
    $ \hat{m}_t = \frac{m_t}{1 - \beta_1^t} $
    $ \hat{v}_t = \frac{v_t}{1 - \beta_2^t} $
4.  **Update parameters:**
    $ \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t + \epsilon}} \odot \hat{m}_t $

in:
*   $\theta_t$ is a parameter at time step $t$.
*   $\eta$ is the global learning rate.
*   $g_t$ is the gradient calculated at time step $t$.
*   $m_t$ and $v_t$ are estimates of the first and second moments respectively.
*   $\beta_1$ and $\beta_2$ are the decay rates (usually set to $\beta_1=0.9$, $\beta_2=0.999$).
*   $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected estimates.
*   $\epsilon$ is a small constant (e.g. $10^{-8}$).
*   $\odot$ represents element-wise multiplication.

#### advantage

*   **Adaptive learning rate and momentum:** Combines the advantages of the two methods, with fast convergence speed and strong robustness.
*   **Insensitive to hyperparameters:** Default parameters usually work well, reducing the burden of parameter adjustment.
*   **Applies to sparse gradients and non-stationary targets. **
*   **Computationally efficient with moderate memory requirements. **

#### shortcoming

*   **May converge to suboptimal solutions:** In some cases, the generalization ability may not be as good as SGD.
*   **Memory Consumption:** Requires more memory than SGD and AdaGrad to store first and second moment estimates.

### Summary and comparison

| Algorithm | Main features | Advantages | Disadvantages |
| :-------- | :----------------------------------------- | :------------------------------------------------------------------- | :------------------------------------------------------------------- |
| **SGD** | Uses single sample update, gradient oscillation | Fast calculation, small memory usage, may jump out of local optimum | Convergence oscillation, sensitive to learning rate, unfriendly to sparse data |
| **AdaGrad** | Cumulative squared gradient, adaptive learning rate | Friendly to sparse gradients, no need to manually adjust the learning rate | The learning rate decays too fast and may stop learning |
| **RMSProp** | Exponential decay average of square gradient, adaptive learning rate | Solve the problem of AdaGrad learning rate decaying too fast, more robust to non-stationary targets | Still need to adjust the learning rate, no momentum |
| **Adam** | First-order moment (momentum) + second-order moment (RMSProp) + deviation correction | Fast convergence, strong robustness, insensitive to hyperparameters, suitable for sparse gradients and non-stationary objectives | May converge to suboptimal solutions, memory consumption is larger than SGD |

In practical applications, Adam is often the default choice for many deep learning tasks because it provides good performance and fast convergence in most cases. However, understanding the principles, advantages and disadvantages of other algorithms can help make better choices in specific scenarios.


