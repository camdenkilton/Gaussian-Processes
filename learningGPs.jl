### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ e844befc-6007-45fd-9fc8-73e3977e72dd
using PlutoUI, LaTeXStrings, Random, Interpolations

# ╔═╡ a84d6c90-9598-11f1-9983-4fc186dd632a
md"""
# Learning Gaussian Processes

This notebook can be read on its own, but it is intended to be a companion to sections 2.2 and 2.3 of [Rassmusen and Williams 2006, *Gaussian Processes for Machine Learning*](https://gaussianprocess.org/gpml/).

C. E. Rasmussen & C. K. I. Williams, *Gaussian Processes for Machine Learning*, the MIT Press, 2006, ISBN 026218253X. © 2006 Massachusetts Institute of Technology. www.GaussianProcess.org/gpml

Other great resources include:
- [A Visual Exploration of Gaussian Processes](https://distill.pub/2019/visual-exploration-gaussian-processes/)
- [Understanding Gaussian Processes: From Theory to Applications](https://franzesegiovanni.github.io/aboutme/files/Gaussian_Process_Lecture_notes.pdf)

"""

# ╔═╡ d274402c-9abc-4d33-84ba-ec536035c29c
md"""
#### What is a Gaussian process?
A Gaussian process (GP) is a probability distribution over functions.
A GP is uniquely specified by its **mean** and **covariance kernel**. A sample, $f(x)$, from a GP is denoted:
```math
f(x) \sim \mathcal{GP} \big( \mu(x), \, \kappa(x, x^\prime) \big)
```
- The mean, $\mu$, is the expected value of the output function at the point $x$, i.e. 
```math
\mu(x) := \mathbb{E}[f(x)]. 
```
- The kernel, $\kappa(x, x^\prime)$, measures the relation (covariance) between the output function values at two points, i.e. 
```math
    \kappa(x, x^\prime)  := \mathbb{E} \Big[ \big( f(x) - \mu(x) \big) \big( f(x^\prime) - \mu(x^\prime) \big) \Big]
     = \text{Cov} (f(x), \, f(x^\prime)).
```

"""

# ╔═╡ 767c6592-dbad-417c-a8e9-e2793f88a684
md"""
#### Why use Gaussian processes?

- Nonparametric framework for inference: GPs depend on **hyperparameters** instead of parameters. There is no fixed functional form underlying the predictive model, allowing it to grow in complexity with a data set.
- Natural uncertanty quantification (UQ): GPs come with a built-in confidence interval at every point in a prediction. This confidence interval does not require generating a large number of samples, as is often the case e.g. with stochastic processes.

"""

# ╔═╡ e955a7e9-766b-4ac1-a759-cfcb18118d78
md"""
!!! tip "Tips for using this notebook"
	- Sliders are meant to be slid. Vary the number of data points, hyperparameters, errors, etc. while observing various plots.
	- Semicolons are used to hide the outputs of cells. Remove them to see information about the object type or data structure.
	- Restarting the notebook takes all interactables back to their defaults.
	- Control click on a function or array if you want to see where it's defined.
"""

# ╔═╡ ab1ccc85-54b2-4a4c-8c80-8ada4a4ace7b
md"""
---
"""

# ╔═╡ 86fecf9b-a0b8-401b-b9b1-0f10bf352d2a
md"""
#### Temporary
"""

# ╔═╡ 35e55246-5965-45e3-98e6-71985c992768
md"""
!!! danger "Questions"
	- Am I using the words "prior" and "posterior" correctly? They're not defined in Rassmusen & Williams, and this and other sources seem to use them slightly differently from how I do here
	- Why use the Cholesky decomposition at all instead of the original covariance matrix?
	- Would it make more sense (mathematically) for changing the hypers and mean with the sliders to also recompute the random vector(s)?
	- Is there a preferred optimization algorithm for maximizing the log marginal likelihood? This seems to be discussed around p.126 of Rassmusen & Williams, but looks like an unnecessarily complicated implementation
	

!!! danger "To do"
	- Add explanations to posterior section
	- Add hyperparameter optimization section
	- Implement other kernels
"""

# ╔═╡ d9dd4992-495d-456f-bf6f-04f7ce57daf1
md"""
---
"""

# ╔═╡ 57e54104-ac1f-4b5b-a6bc-b7a465f88fa5
md"""
Let's first take a look at the example data that we will be conditioning a GP to. This data represents the set of observations or calculations that we'd like our model to be aware of, in order to make better predictions. In this notebook, we will be conditioning a GP to a set of data of the form,
```math
f(x) = x \sin(x).
```
"""

# ╔═╡ 0ae78386-af1d-4baf-8e6d-ffe7e029eb2f
func(x) = x .* sin.(x);

# ╔═╡ 490fd2e9-bfa2-4f19-b67d-44c076060a87
begin
	# Construct the example data set
	x_data = LinRange(-2, 7, 10)
	y_data = func(x_data)
	
	# Define the domain on which to evaluate GP samples
	x_grid = LinRange(minimum(x_data)-1, maximum(x_data)+1, 300)
end;

# ╔═╡ 0292bb3f-72d7-43e9-b50d-ed7901bbc259
md"""
The training data set is referred to as $(X, \, Y)$, and $\texttt{x\_grid}$ is referred to as $X_*$.
"""

# ╔═╡ c13024f9-8ff5-41fd-9601-54213a65022c
md"""
We'll begin in the simplest case:
- Perfectly precise and accurate training data
- A squared exponential (SE) kernel
- Zero mean function: $\mu(x) = 0$
Later, we'll explore the effects of modifying all of these points.
"""

# ╔═╡ bb83c630-c983-4555-8736-078c235d087d


# ╔═╡ 424ea1cc-3796-471e-84a2-f3b1b2e70f93
md"""
## Creating the prior
"""

# ╔═╡ ab633f5f-18f8-474a-8b0a-b5e2d3e900b3
md"""
The SE kernel takes the form,
```math
\text{cov} \big( f(x_1), f(x_2) \big) = 
\kappa (x_1, x_2) = 
\sigma^2 \exp \left( \frac{-|x_1 - x_2|^2}{2l^2} \right).
```
!!! info "Note"
	The hyperparameters, $\theta = (\sigma, l)$, represent the signal variance and length scale respectively.

"""

# ╔═╡ d370e276-817c-4fae-9bf6-6211521f48e4
kern_SE(x₁, x₂; σ = 1, l = 1) = σ^2 * exp(- (abs(x₁-x₂)^2) / (2*l^2));

# ╔═╡ 1aa5ad5e-7cd5-4936-bdd5-436005bd5f1d
μ(x) = 0;

# ╔═╡ 2d98a4d8-1bad-4d55-9ffa-c8f5b07b1d91
md"""
As I pointed out earlier, these two functions (the kernel and mean) sufficiently and uniquely define a GP. Let's look at some samples from this **prior**.
!!! info "Definition"
	A **prior** refers to the probability distribution *before* it is conditioned on data.
	
"""

# ╔═╡ 293a0766-d890-41b2-b20d-649f8a417723
md"""
The prior covariance, $K(X_*, \, X_*)$, contains the kernel evaluations at every pair of points in the domain of the GP samples, making it $n \times n$. Note that this $n$ is the length of $\texttt{x\_grid}$; generally much larger than the length of $\texttt{x\_data}$.
```math
K_{ij} = \kappa(x_i, x_j)
```
"""

# ╔═╡ 9da93f79-a4de-42ae-86d2-8f1a643d1a86
md"""
The covariance *matrix* is symmetric because the covariance *function* is symmetric:
```math
K_{ij} = \text{cov} \big( f(x_i), f(x_j) \big) = \text{cov} \big( f(x_j), f(x_i) \big) = K_{ji}.
```
Additionally, this matrix is positive definite. The Cholesky decomposition is numerically stable and computationally efficient, and applies to symmetric, positive definite matrices. Since the covariance matrix is strictly only positive semidefinite in theory, and numerical errors are inevitable, a small error term must be added to the diagonal of the covariance matrix to enforce nonsingularity.
"""

# ╔═╡ 3ebdaf2c-b4fd-40d2-82d7-2f3453db6aaa
md"""
To generate a particular sample from the prior, we generate a random vector, $z$, of length $n$. Each entry of this vector is a sample from a 1-D Gaussian, $\mathcal{N}(0, 1)$. 
"""

# ╔═╡ c196c6e9-ebbf-4d8d-8971-55ab996a7b75
# Evaluate the mean function for each point in the domain
μ_grid = μ.(x_grid);

# ╔═╡ fee8af7c-e5be-4762-bd42-320ee6409c5d
md"""
The random vector becomes a sample from our particular prior via the (decomposed) covariance matrix.

```math
f_\text{prior} = \mu_\text{prior} + L_\text{prior} \, z \ \sim \ \mathcal{N}(\mu_\text{prior}, \, K(X_*, \, X_*))
```
"""

# ╔═╡ 340ce1ff-94d1-49c7-8097-abbace12277e
md"""
**σ** = $(@bind σ_prior Slider(0.04:0.04:3.0, default=1.0, show_value=true))

**L** = $(@bind l_prior Slider(0.04:0.04:2.0, default=1.0, show_value=true))
"""

# ╔═╡ dd104e69-632f-4df8-ba5d-486dcad33e76
# Construct the covariance matrix
K_prior = [kern_SE(x_grid[i], x_grid[j]; σ = σ_prior, l = l_prior) for i in eachindex(x_grid), j in eachindex(x_grid)];

# ╔═╡ add21cef-f53d-4669-90fa-f8c9c5726eea
@bind new1 CounterButton("Click for a new sample")

# ╔═╡ 1ccd9592-b87a-4358-bb9d-f03c06c30992
begin
	new1 	# Allows for sampling via GUI button press
	z = randn(length(x_grid)) 	# Generate random vector
end

# ╔═╡ be1bdb7a-27b3-4412-ac90-b28bd602723d
md"""
Since the GP prior is completely unaware of the data, these samples are (of course) usually nowhere close to following the data. 

**Show data**: $(@bind show_dat1 Switch(default=false))


For this reason, a prior on its own is generally not very powerful. This is why we might want to **condition** the GP on the data set, so we can create a **posterior**.
"""

# ╔═╡ a90e0991-87f1-4acb-b09e-21cc629effbc


# ╔═╡ f68db00f-bc14-4855-8e92-e6c802b92d1f
md"""
## Creating the posterior
"""

# ╔═╡ 43f4dfc5-fff8-4aca-ad55-b1ed67f3bbcd
md"""
!!! info "Definition"
	A **posterior** refers to the probability distribution *after* it is conditioned on data.
	
"""

# ╔═╡ 2748fe80-182a-40f5-a23c-0b0124d85f3c
md"""
The posterior covariance is the same as the prior covariance matrix with the addition of the submatrices consisting of the kernel evaluations that include the data set:
```math
\begin{bmatrix}
K(X, \, X) & K(X, \, X_*) \\
K(X_*, \, X) & K(X_*, \, X_*)
\end{bmatrix}
```
"""

# ╔═╡ 62b6b050-a7dc-4b1e-9347-a2e161a80f31
μ_data = μ.(x_data);

# ╔═╡ 7b6030dd-837f-42f1-b29c-f1843d6a3f38
md"""
I don't know how to explain this
"""

# ╔═╡ a9921f35-b0f5-42d6-b83e-792bb2bab518
md"""
I don't know how to explain this
"""

# ╔═╡ 09e7242c-ce2a-48a9-a2cb-1c909c1a0215
md"""
With the posterior/predictive mean and covariance, the sample is created in the same way as with the prior.
```math
 ? = \mu_\text{posterior} + L_\text{posterior} \, z \ \sim \ \mathcal{N}\left(\mu_\text{posterior}, \, 
\begin{bmatrix}
K(X, \, X) & K(X, \, X_*) \\
K(X_*, \, X) & K(X_*, \, X_*)
\end{bmatrix}
\right)
```
"""

# ╔═╡ 683f5da3-08a8-48e9-872b-90e56cb263be
md"""
**σ** = $(@bind σ_post Slider(0.02:0.02:3.0, default=1.0, show_value=true))

**L** = $(@bind l_post Slider(0.04:0.02:2.0, default=1.0, show_value=true))

**Number of data points**: $(@bind data_len Slider(3:1:25; default = 8, show_value = true))
ㅤㅤㅤ**Show data**: $(@bind show_dat2 Switch(default=true))
"""

# ╔═╡ e9cd5c6b-a345-40c5-a32d-77bb7afc2002
begin
	K_data = [kern_SE(x_data[i], x_data[j]; σ = σ_post, l = l_post) for i in eachindex(x_data), j in eachindex(x_data)]
    
    K_grid_data = [kern_SE(x_grid[i], x_data[j]; σ = σ_post, l = l_post) for i in eachindex(x_grid), j in eachindex(x_data)]

    K_grid = [kern_SE(x_grid[i], x_grid[j]; σ = σ_post, l = l_post) for i in eachindex(x_grid), j in eachindex(x_grid)]
end;

# ╔═╡ c06100c0-5469-4b8d-848e-ba814731bbb3
α = K_data \ (y_data - μ_data);

# ╔═╡ 411f4557-8e91-4590-9981-9807f4252fb8
μ_post = μ_grid + K_grid_data * α;

# ╔═╡ 6ac5d838-cbfd-4863-87f7-fa119ef223cb
md"""
**Show the true function**: $(@bind show_func2 Switch(default=false))
"""

# ╔═╡ 8e464c2b-2327-4726-a790-fc1465c074b7
@bind new2 CounterButton("New sample")

# ╔═╡ 793d30ea-aa71-426b-99c9-85a99a496105
begin
	new2
	z_post = randn(length(x_grid))
end;

# ╔═╡ 52dc6f96-0f95-434e-803c-bddd4f8828f8
md"""
Notice that since the error on the data is zero, any posterior sample will pass directly through every data point. In the next section, we'll see the effect of adding error bars to the data.
"""

# ╔═╡ 9ebac8ef-52b3-49b8-b183-e37e8aa2bdc5


# ╔═╡ df76cccc-f606-4eb0-8470-bd90acfffdd1
md"""
## But what about uncertanty quantification?
"""

# ╔═╡ 56571ca9-494a-4bcb-a04b-76fc37db0bda
md"""
Let's look at a large ensemble of samples to get a sense of the uncertainties we're dealing with.
"""

# ╔═╡ 9067ecad-4ff8-446d-9253-52dfe1f87280
md"""
**Number of samples**: $(@bind N_ens Slider(25:25:400, default=75, show_value = true))
"""

# ╔═╡ 5a20613e-04b4-4c6b-958c-cf89d9374453
@bind new_ens CounterButton("New ensemble")

# ╔═╡ d36d8b0d-3690-403d-8d68-a48e371bd26f
md"""
**σ** = $(@bind σ_ens Slider(0.02:0.02:4.0, default=1.0, show_value=true))
ㅤ**L** = $(@bind l_ens Slider(0.1:0.02:2.0, default=1.0, show_value=true))
ㅤ**Constant mean** = $(@bind mean_ens Slider(0.0:0.08:8.0, default=0.0, show_value=true))
"""

# ╔═╡ 6ffc4d87-4c05-47f2-b525-a490853e8e80
md"""
Let's try changing the training data:

**Remove a training point**: $(@bind rem_dat Switch(default=false))
ㅤIndex of point to be removed: $(@bind rem_idx Slider(1:1:length(x_data), default = 4, show_value =true))

**Number of data points**: $(@bind data_len_ens Slider(3:1:20; default = 8, show_value = true))

**Error on data**: $(@bind error_ens Slider(0.0:0.01:1.5, default = 0.0, show_value = true))
"""

# ╔═╡ b98b9269-7513-48ba-8c95-2bdb3b97fc13
md"""
**Show the true function**: $(@bind show_func3 Switch(default=true))
"""

# ╔═╡ bb614bfd-8880-43ec-9af4-f719385d7ff5
md"""
One of the interesting qualities of GPs is how we can plot the mean and confidence intervals without generating any samples. By definition, the average of an infinite number of samples is $\mu(x)$. The confidence interval is...
"""

# ╔═╡ e6c1dd95-527f-4d5e-ad31-7189d915d3d4
md"""
## How to choose (optimize) hyperparameters
"""

# ╔═╡ 358ad43b-becb-4a7b-9856-2dcc764b398c


# ╔═╡ 287849c4-7960-41e3-951e-9d604ae615f0


# ╔═╡ 54ab515d-9917-4029-b2da-d682a6e715eb
md"""
## Other kernels and mean functions
"""

# ╔═╡ f018dd0d-4d31-4c13-aa0c-0193fcbe8ff3


# ╔═╡ 7043ffb5-26ec-4847-821b-4d25a52dc409


# ╔═╡ b828b38a-8e2d-482d-8b4c-156bf39aef0f


# ╔═╡ 306cb82b-252a-45ee-b6b7-886c3da200d6
md"""
---
"""

# ╔═╡ f09fcc93-3176-48ec-a43f-526a9607eae0
md"""
## Extra Code
"""

# ╔═╡ c8a3ed03-6658-47e7-a9ac-e73ff43c168f
md"""
#### Helper functions
"""

# ╔═╡ d3d1349d-89c3-4947-88a4-9e6849cb33c0
md"""
#### UQ section
"""

# ╔═╡ 045667a8-ab00-4151-947c-33a4eb210f29
kern_SE_ens(x₁, x₂; σ = σ_ens, l = l_ens) =
	σ^2 * exp(- (abs(x₁-x₂)^2) / (2*l^2));

# ╔═╡ 3b0e0f3b-71ac-405c-be19-48566b4b627d
μ_ens(x) = mean_ens;

# ╔═╡ cae3005f-a6c6-4eb1-9f77-ea6c9207be5b
md"""
Generate ensemble from prior:
"""

# ╔═╡ 286a2938-96ad-4c6b-97b0-37aecc463961
md"""
Generate ensemble from posterior:
"""

# ╔═╡ 0286bd4a-3804-49a7-8769-efc14100fdf9
x_data_ens = LinRange(-2, 7, data_len_ens);

# ╔═╡ e742d28b-154b-496c-ac11-d819a87d0619
y_data_ens = func(x_data_ens);

# ╔═╡ 2019e935-14c5-4fd2-8891-6f8de67828a2
md"""
Handle removing a data point:
"""

# ╔═╡ 0cd3e387-8c4e-43af-840f-e5abd5a701a1
md"""
Calculate... for ribbon plots:
"""

# ╔═╡ 1f157e3a-b56e-4fb5-b64a-0306fe0ca6a5


# ╔═╡ e5510571-9685-4dbc-be7e-56f0754091ca
md"""
---
"""

# ╔═╡ 7820e1d5-191b-45d8-9548-b1daff296216
begin
	import Plots as plt
	import LinearAlgebra as la
end

# ╔═╡ b46df146-81d4-4092-90ba-429a9d0740c9
begin
	# Initialize the plot
	plt.plot(xlabel = L"x", ylabel = L"y",
		dpi = 200, framestyle =:box)
	
	# Add the true function
	plt.plot!(
		x_grid,
		func(x_grid),
		label = L"f(x)")
	
	# Add the example data
	plt.scatter!([x_data], [y_data], label = "Data")
end

# ╔═╡ fe497409-53f4-4b47-9672-4e717d077b02
L_prior = la.cholesky( la.Symmetric(K_prior + 1e-10*la.I) ).L;

# ╔═╡ 59a5ab1e-efea-4ef5-aef4-9a28a583921e
f_prior = μ_grid + L_prior * z;

# ╔═╡ b89d59c8-870d-4aaa-ac7b-8b8a79d8fd2b
begin
	# Initialize plot
	p1 = plt.plot(xlabel = L"x", ylabel = L"y", title = "Prior sample",
		dpi = 200, framestyle =:box, gridalpha = 0.05)
	# Plot the sample
	plt.plot!(x_grid, f_prior, lw =1.5, label = "GP prior sample")
	# Toggle data
	if show_dat1
		plt.scatter!([x_data], [y_data], label = "Data")
		plt.plot!(x_grid, func(x_grid), linestyle = :dash, lc = :black,
						   label = "True function")
	else
	plt.plot!()
	end
end

# ╔═╡ 07366c09-22d3-4c01-9e89-c3e57a59d00e
K_post = la.Symmetric(K_grid - K_grid_data * (K_data \ K_grid_data'));

# ╔═╡ 274a1335-f51a-4dbf-b4c6-7f6d073fd041
L_post = la.cholesky(K_post + 1e-10 * la.I).L;

# ╔═╡ 545f5bcf-ac36-403f-9436-83757658d770
f_post = μ_post + L_post * z_post;

# ╔═╡ cbe8eba8-8fe6-4f8d-ab14-29bac4db8684
begin
	# Initialize plot
	p_post = plt.plot(xlabel = L"x", ylabel = L"y", title = "Posterior sample",
		dpi = 200, framestyle =:box, gridalpha = 0.05)
	# Plot the sample
	plt.plot!(x_grid, f_post, lw =1.5, label = "GP posterior sample")
	# Toggle data
	show_dat2 ? plt.scatter!(x_data, y_data, label = "Data") : plt.plot!()
	# Toggle true function
	show_func2 ? plt.plot!(x_grid, func(x_grid), linestyle = :dash, lc = :black,
						   label = "True function") : plt.plot!()

	# Calculate and plot residuals
	f_post_interp = CubicSplineInterpolation(x_grid, f_post)
	res_post = y_data - f_post_interp(x_data)
	res_post_full = func(x_grid) - f_post
	
	p_post_resi = plt.plot(x_grid, res_post_full, label = "Residual",
		dpi = 200, framestyle =:box, gridalpha = 0.03)
	show_dat2 ? plt.scatter!(p_post_resi, x_data, res_post, label = "",xlabel = L"x") : plt.plot!()
	plt.hline!(p_post_resi, [0], lc =:black, lw = 2, label = "")

	# Create full plot with both subplots
	layout = plt.@layout([a{0.75h}; b{0.25h}])
	plt.plot(p_post, p_post_resi;
    layout = layout, link = :x, size = (700, 600), dpi = 200)
end

# ╔═╡ b52ddf87-e64a-431c-a855-9adbe10fec75
# 	prior(x_grid, mean, kern; N = 1)

# Sample from a prior. x_grid is the domain of the GP samples, N is the number of samples
function prior(x_grid, mean, kern; N = 1)


	# Construct covariance matrix
	K = [kern(x_grid[i], x_grid[j]) for i in eachindex(x_grid), j in eachindex(x_grid)]

	# Cholesky decomposition
	L = la.cholesky( la.Symmetric(K + 1e-8*la.I) ).L

	# Generate random vector(s)
	z = randn(length(x_grid), N)

	# Evaluate the mean function
	μ_grid = mean.(x_grid)

	# Construct
	f = μ_grid .+ L * z
	
	return f
end

# ╔═╡ 1875f6d4-f9e4-4013-bcc9-33466b4ad3db
begin
	new_ens
	F_prior = prior(x_grid, μ_ens, kern_SE_ens, N = N_ens)
end;

# ╔═╡ c47e5230-3442-42c6-8b8a-2b47e7a0a5cc
begin
    p_prior_ens = plt.plot(xlabel = L"x", ylabel = L"y", title = "Prior ensemble", 
                           dpi = 300, framestyle =:box, gridalpha = 0.05)
    for i in 1:N_ens
        plt.plot!(x_grid, F_prior[:, i], alpha=0.5, label = "")
    end
    p_prior_ens
end

# ╔═╡ 2b7688cc-1a23-45f8-9331-04588143314d
# Sample from a posterior. x_grid is the domain of the GP samples, N is the number of samples
function posterior(x_dat, y_dat, error, x_grid, mean, kern; N = 1)

	# Construct covariance matrices
	K_data_helper = [
        kern(x_dat[i], x_dat[j]) for i in eachindex(x_dat), j in eachindex(x_dat)
    ] + error^2 * la.I;

    K_grid_helper = [
        kern(x_grid[i], x_grid[j]) for i in eachindex(x_grid), j in eachindex(x_grid)];
    
    K_gd_helper = [
        kern(x_grid[i], x_dat[j])
        for i in eachindex(x_grid), j in eachindex(x_dat)
    ]

    μ_data_helper = mean.(x_dat)

    α_helper = K_data_helper \ (y_dat - μ_data_helper)

    μ_grid_helper = mean.(x_grid)
    μ_post_helper = μ_grid_helper + K_gd_helper * α_helper

    K_post_helper = la.Symmetric(K_grid_helper - K_gd_helper * (K_data_helper \ K_gd_helper'))

	# Cholesky decomposition
	L_post_helper = la.cholesky(K_post_helper + 1e-8 * la.I).L

	# Generate random vector(s)
	z_post_helper = randn(length(x_grid), N)

	# Construct
	f_post_helper = μ_post_helper .+ L_post_helper * z_post_helper

    # 
    std_helper = sqrt.(la.diag(K_post_helper+ 1e-8 * la.I))
	
	return f_post_helper, μ_post_helper, std_helper
end

# ╔═╡ ea9a22f0-08b3-48ad-9bbb-b490bfe6cdef
begin
	new_ens
	F_post_ens, μ_post_ens, std_post_ens = posterior(x_data_ens, y_data_ens,
				   error_ens, x_grid, μ_ens, kern_SE_ens, N = N_ens)
end;

# ╔═╡ 654f77a0-afd5-468c-9a2a-76fb8ed33240
if rem_dat
	x_rem = x_data_ens[[i for i in eachindex(x_data_ens) if i != rem_idx]]
	y_rem = y_data_ens[[i for i in eachindex(y_data_ens) if i != rem_idx]]
	F_post_rem, μ_post_rem, std_post_rem = posterior(x_rem, y_rem,
				   error_ens, x_grid, μ_ens, kern_SE_ens, N = N_ens)
end;

# ╔═╡ 7a0614a4-f604-463d-8aef-62867891069c
begin
    p_post_ens = plt.plot(xlabel = L"x", ylabel = L"y", title = "Posterior ensemble", 
                           dpi = 300, framestyle =:box, gridalpha = 0.05)
    if rem_dat
        for i in 1:N_ens
            plt.plot!(x_grid, F_post_rem[:, i], lw = 0.3, alpha=0.8, label = "")
        end
        plt.scatter!(x_rem, y_rem, yerror = error_ens, mc = :black, label = "Data")
        plt.scatter!([x_data_ens[rem_idx]], [y_data_ens[rem_idx]], mc = :red, label = "")
    else
        for i in 1:N_ens
            plt.plot!(x_grid, F_post_ens[:, i], lw = 0.3, alpha=0.8, label = "")
        end
        plt.scatter!(x_data_ens, y_data_ens, yerror = error_ens, mc = :black, label = "Data")
    end
    
    # Toggle true function
	show_func3 ? plt.plot!(x_grid, func(x_grid), linestyle = :dash, lc = :black,
						   label = "True function") : plt.plot!()
    p_post_ens
end

# ╔═╡ 1840b1d9-12e8-4c63-b7c7-b3388cd09baf
begin
    if rem_dat
        plt.plot(x_grid, μ_post_rem;
            lw = 2, label = L"Posterior mean $\pm 1\sigma$", xlabel = L"x", ylabel = L"y",
            title = "UQ - Posterior",
            dpi = 300, framestyle = :box, gridalpha = 0.05,
            ribbon = 1.96 .* std_post_rem)
        plt.scatter!(x_rem, y_rem, yerror = error_ens, mc = :black, label = "Data")
        plt.scatter!([x_data_ens[rem_idx]], [y_data_ens[rem_idx]], mc = :red, label = "")
    else
        plt.plot(x_grid, μ_post_ens;
            lw = 2, label = L"Posterior mean $\pm 1\sigma$", xlabel = L"x", ylabel = L"y",
            title = "UQ - Posterior",
            dpi = 300, framestyle = :box, gridalpha = 0.05,
            ribbon = 1.96 .* std_post_ens)
        plt.scatter!(x_data_ens, y_data_ens, yerror = error_ens, mc = :black, label = "Data")
    end
    
    # Toggle true function
	show_func3 ? plt.plot!(x_grid, func(x_grid), linestyle = :dash, lc = :black,
						   label = "True function") : plt.plot!()
end

# ╔═╡ 44d468ce-ea36-441e-84d5-67e0a0b5ba1b
prior_std = sqrt.(la.diag(K_prior));

# ╔═╡ d43677d5-1f27-4278-8d53-e9e53d6efac2
begin
    plt.plot(x_grid, μ_ens.(x_grid); lw = 2,
        label = L"Prior mean $\pm 1\sigma$", xlabel = L"x", ylabel = L"y", title = "UQ - Prior",
        dpi = 300, framestyle = :box, gridalpha = 0.05,
        ribbon = 1.96 .* prior_std)
    # plt.scatter!(x_data_ens, y_data_ens, yerror = error_ens, mc = :black, label = "Data")
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Interpolations = "~0.16.3"
LaTeXStrings = "~1.4.0"
Plots = "~1.41.6"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "27002a1aebf4f68eb7e04982a21da5db5393f1ec"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "6c3913f4e9bdf6ba3c08041a446fb1332716cbc2"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.0"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "daa72978cd7a624246e894a4f4f067706d4e17e2"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "bbe1079eecf9c9fbb52765193ad2bae27ae09bc8"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.10"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "3c9be947934c38475bafe822c6d61aaed17f0738"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.6.0"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e6c4a6407a949e79a9d3f249bf49e6987c80e01f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.2+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "7a58e45171b63ed4782f2d36fdee8713a469e6e0"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.2+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "9e0fb9e54594c47f278d75063980e43066e26e20"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "f954322d5de03ec630d177cda203dcd92b6be399"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.26"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "6fada551286ab6ea4ca1628cb2de9f166a2ec966"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.26+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "090526e65de8f69648ac156daae153de8b56df62"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.88.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "51059d23c8bb67911a2e6fd5130229113735fc7e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.11.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "48922d06068130f87e43edef52382e6a94305ae6"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.3"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c89d196f5ffb64bfbf80985b699ea913b0d2c211"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.6.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1dae3057da6f2b9c857afef03177bbdc7c4afe92"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b7970cef8ae1c990ba0c09cd8bdc1145e006632f"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "22.1.7+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "24390f715ff0795a1c4b912d788f18c52c6abd19"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.11"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "8785729fa736197687541f7053f6d8ab7fc44f92"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.10"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7126b66b721a605a2fec966a2874c5ed53258eb3"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "3de8f5e6e90ebfa8d6d1f86997d6cdcd6a912ff3"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.7"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "cb20a4eacda080e517e4deb9cfb6c7c518131265"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.6"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "246a8bb2e6667f832eea063c3a56aef96429a3db"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.18"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "e4d7a1a0edc20af42689ea6f4f3587a2175d50ee"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.12"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "82bee338d650aa515f31866c460cb7e3bcef90b8"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.2"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "5253f44481f18cd938d4559d5e44fa82198408a6"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.3"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "dcb316b3ce0941f195537dda56bea4517fcd3ff5"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.4+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "2e59214e017a55cb87474a00fa76035c82ac0e17"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+2"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "850b06095ee71f0135d644ffd8a52850699581ed"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╟─a84d6c90-9598-11f1-9983-4fc186dd632a
# ╟─d274402c-9abc-4d33-84ba-ec536035c29c
# ╟─767c6592-dbad-417c-a8e9-e2793f88a684
# ╟─e955a7e9-766b-4ac1-a759-cfcb18118d78
# ╟─ab1ccc85-54b2-4a4c-8c80-8ada4a4ace7b
# ╟─86fecf9b-a0b8-401b-b9b1-0f10bf352d2a
# ╟─35e55246-5965-45e3-98e6-71985c992768
# ╟─d9dd4992-495d-456f-bf6f-04f7ce57daf1
# ╟─57e54104-ac1f-4b5b-a6bc-b7a465f88fa5
# ╠═0ae78386-af1d-4baf-8e6d-ffe7e029eb2f
# ╟─b46df146-81d4-4092-90ba-429a9d0740c9
# ╠═490fd2e9-bfa2-4f19-b67d-44c076060a87
# ╟─0292bb3f-72d7-43e9-b50d-ed7901bbc259
# ╟─c13024f9-8ff5-41fd-9601-54213a65022c
# ╟─bb83c630-c983-4555-8736-078c235d087d
# ╟─424ea1cc-3796-471e-84a2-f3b1b2e70f93
# ╟─ab633f5f-18f8-474a-8b0a-b5e2d3e900b3
# ╠═d370e276-817c-4fae-9bf6-6211521f48e4
# ╠═1aa5ad5e-7cd5-4936-bdd5-436005bd5f1d
# ╟─2d98a4d8-1bad-4d55-9ffa-c8f5b07b1d91
# ╟─293a0766-d890-41b2-b20d-649f8a417723
# ╠═dd104e69-632f-4df8-ba5d-486dcad33e76
# ╟─9da93f79-a4de-42ae-86d2-8f1a643d1a86
# ╠═fe497409-53f4-4b47-9672-4e717d077b02
# ╟─3ebdaf2c-b4fd-40d2-82d7-2f3453db6aaa
# ╠═1ccd9592-b87a-4358-bb9d-f03c06c30992
# ╠═c196c6e9-ebbf-4d8d-8971-55ab996a7b75
# ╟─fee8af7c-e5be-4762-bd42-320ee6409c5d
# ╠═59a5ab1e-efea-4ef5-aef4-9a28a583921e
# ╟─b89d59c8-870d-4aaa-ac7b-8b8a79d8fd2b
# ╟─340ce1ff-94d1-49c7-8097-abbace12277e
# ╟─add21cef-f53d-4669-90fa-f8c9c5726eea
# ╟─be1bdb7a-27b3-4412-ac90-b28bd602723d
# ╟─a90e0991-87f1-4acb-b09e-21cc629effbc
# ╟─f68db00f-bc14-4855-8e92-e6c802b92d1f
# ╟─43f4dfc5-fff8-4aca-ad55-b1ed67f3bbcd
# ╟─2748fe80-182a-40f5-a23c-0b0124d85f3c
# ╠═e9cd5c6b-a345-40c5-a32d-77bb7afc2002
# ╠═62b6b050-a7dc-4b1e-9347-a2e161a80f31
# ╟─7b6030dd-837f-42f1-b29c-f1843d6a3f38
# ╠═c06100c0-5469-4b8d-848e-ba814731bbb3
# ╠═411f4557-8e91-4590-9981-9807f4252fb8
# ╟─a9921f35-b0f5-42d6-b83e-792bb2bab518
# ╠═07366c09-22d3-4c01-9e89-c3e57a59d00e
# ╠═274a1335-f51a-4dbf-b4c6-7f6d073fd041
# ╟─09e7242c-ce2a-48a9-a2cb-1c909c1a0215
# ╠═793d30ea-aa71-426b-99c9-85a99a496105
# ╠═545f5bcf-ac36-403f-9436-83757658d770
# ╟─cbe8eba8-8fe6-4f8d-ab14-29bac4db8684
# ╟─683f5da3-08a8-48e9-872b-90e56cb263be
# ╟─6ac5d838-cbfd-4863-87f7-fa119ef223cb
# ╟─8e464c2b-2327-4726-a790-fc1465c074b7
# ╟─52dc6f96-0f95-434e-803c-bddd4f8828f8
# ╟─9ebac8ef-52b3-49b8-b183-e37e8aa2bdc5
# ╟─df76cccc-f606-4eb0-8470-bd90acfffdd1
# ╟─56571ca9-494a-4bcb-a04b-76fc37db0bda
# ╟─9067ecad-4ff8-446d-9253-52dfe1f87280
# ╟─5a20613e-04b4-4c6b-958c-cf89d9374453
# ╟─c47e5230-3442-42c6-8b8a-2b47e7a0a5cc
# ╟─7a0614a4-f604-463d-8aef-62867891069c
# ╟─d36d8b0d-3690-403d-8d68-a48e371bd26f
# ╟─6ffc4d87-4c05-47f2-b525-a490853e8e80
# ╟─b98b9269-7513-48ba-8c95-2bdb3b97fc13
# ╟─bb614bfd-8880-43ec-9af4-f719385d7ff5
# ╟─1840b1d9-12e8-4c63-b7c7-b3388cd09baf
# ╟─d43677d5-1f27-4278-8d53-e9e53d6efac2
# ╟─e6c1dd95-527f-4d5e-ad31-7189d915d3d4
# ╠═358ad43b-becb-4a7b-9856-2dcc764b398c
# ╠═287849c4-7960-41e3-951e-9d604ae615f0
# ╟─54ab515d-9917-4029-b2da-d682a6e715eb
# ╠═f018dd0d-4d31-4c13-aa0c-0193fcbe8ff3
# ╠═7043ffb5-26ec-4847-821b-4d25a52dc409
# ╟─b828b38a-8e2d-482d-8b4c-156bf39aef0f
# ╟─306cb82b-252a-45ee-b6b7-886c3da200d6
# ╟─f09fcc93-3176-48ec-a43f-526a9607eae0
# ╟─c8a3ed03-6658-47e7-a9ac-e73ff43c168f
# ╟─b52ddf87-e64a-431c-a855-9adbe10fec75
# ╟─2b7688cc-1a23-45f8-9331-04588143314d
# ╟─d3d1349d-89c3-4947-88a4-9e6849cb33c0
# ╠═045667a8-ab00-4151-947c-33a4eb210f29
# ╠═3b0e0f3b-71ac-405c-be19-48566b4b627d
# ╟─cae3005f-a6c6-4eb1-9f77-ea6c9207be5b
# ╠═1875f6d4-f9e4-4013-bcc9-33466b4ad3db
# ╟─286a2938-96ad-4c6b-97b0-37aecc463961
# ╠═0286bd4a-3804-49a7-8769-efc14100fdf9
# ╠═e742d28b-154b-496c-ac11-d819a87d0619
# ╠═ea9a22f0-08b3-48ad-9bbb-b490bfe6cdef
# ╟─2019e935-14c5-4fd2-8891-6f8de67828a2
# ╠═654f77a0-afd5-468c-9a2a-76fb8ed33240
# ╟─0cd3e387-8c4e-43af-840f-e5abd5a701a1
# ╠═44d468ce-ea36-441e-84d5-67e0a0b5ba1b
# ╠═1f157e3a-b56e-4fb5-b64a-0306fe0ca6a5
# ╟─e5510571-9685-4dbc-be7e-56f0754091ca
# ╠═7820e1d5-191b-45d8-9548-b1daff296216
# ╠═e844befc-6007-45fd-9fc8-73e3977e72dd
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
