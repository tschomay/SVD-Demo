# SVD Demonstration - Ted Schomay
# This code demonstrates the use of the singular value decomposition (SVD) for
# feature extraction and dimensionality reduction. It provides simple examples
# on synthetic data.

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure output directory exists for figures
os.makedirs('Figures', exist_ok=True)

# Create data
# We'll start by just creating a single, simple sine function
# Evaluate the sine every 5 degrees for a full period
signal = np.sin(np.array(range(0, 360, 5)) * np.pi / 180.)
plt.plot(signal)
plt.savefig('Figures/1.png', bbox_inches='tight')
plt.clf()

# 1.
# Let's make a matrix of multiple observations of this same pattern
# Here we have 20 observations (columns) of the same signal. Each column
# contains the same sine pattern.
data = np.transpose(np.array([signal for i in range(20)]))

# Now use the SVD to find basis vectors for this dataset. This will find the
# patterns that capture most of the data.
u, s, v = np.linalg.svd(data)

# The diagonal matrix s of singular values shows how prominent each pattern is
# in the data. Let's plot it to see how many patterns we need to capture the
# data
plt.scatter(range(len(s)), s)
plt.xticks(range(len(s)), [x + 1 for x in range(len(s))])
plt.savefig('Figures/2.png', bbox_inches='tight')
plt.clf()

# So the first pattern captures the data completely.
# Next, let's look at the left basis vectors to see what pattern describes the
# columns completely. Plot the first column of u
plt.plot(u[:, 0])
plt.savefig('Figures/3.png', bbox_inches='tight')
plt.clf()

# Plot the second left basis vector to illustrate that it doesn't capture
# anything
plt.plot(u[:, 1])
plt.savefig('Figures/4.png', bbox_inches='tight')
plt.clf()

# As expected, this pattern is non-informative. Now let's look at the
# corresponding right basis vector, or pattern across the rows.
plt.plot(v[0])
plt.savefig('Figures/5.png', bbox_inches='tight')
plt.clf()


# Here it is flat. This shows that there is no varation between the columns.
# 
# 2.
# Next, let's add separate noise to each observation. As a first case, add 
# normally distributed noise with mean 0 and standard deviation 0.1.
mu = 0
sigma = 0.1
noise = np.random.normal(mu, sigma, data.shape)

# Add to data
data_noisy = data + noise

# Have a look at the first two noisy observations
plt.plot(data_noisy[:,0])
plt.plot(data_noisy[:,1])
plt.savefig('Figures/6.png', bbox_inches='tight')
plt.clf()

# Now run SVD on the noisy data
u, s, v = np.linalg.svd(data_noisy)

# This time we will plot the singular values on a log scale because we expect
# most of them to be small, but not zero.
plt.scatter(range(len(s)), s)
plt.xticks(range(len(s)), [x + 1 for x in range(len(s))])
plt.gca().set_yscale('log')
plt.savefig('Figures/7.png', bbox_inches='tight')
plt.clf()

# Now let's check the first basis vector across the rows. This should be
# our sine wave with most of the noise filtered out.
plt.plot(u[:,0])
plt.savefig('Figures/8.png', bbox_inches='tight')
plt.clf()

# The remaining patterns should capture the noise. Let's look at the second one.
plt.plot(u[:,1])
plt.savefig('Figures/9.png', bbox_inches='tight')
plt.clf()

# The first RBV is still essentially flat as it was before  but with a little
# noise added.
plt.plot(v[0])
plt.ylim([-0.5, 0])
plt.savefig('Figures/10.png', bbox_inches='tight')
plt.clf()

# 3.
# Instead of noise, what if we modify the signal observed?
# Let's see what happens if we use orthogonal functions as columns
data_scaled = np.transpose([np.sin(n * np.array(range(0, 360, 2)) * 
                            np.pi / 180.) for n in range(1, 21)])
plt.plot(data_scaled[:,0])
plt.plot(data_scaled[:,1])
plt.plot(data_scaled[:,2])
plt.savefig('Figures/11.png', bbox_inches='tight')
plt.clf()

# Take the SVD
u, s, v = np.linalg.svd(data_scaled)
plt.scatter(range(len(s)), s)
plt.xticks(range(len(s)), [x + 1 for x in range(len(s))])
plt.savefig('Figures/12.png', bbox_inches='tight')
plt.clf()

# Here we see that all singular values have the same value. I.e., we need all
# singular vectors to describe the data. Therefore one possible SVD would be
# the original matrix multiplied by two identities. However this is degenerate
# because all the singular values are equal. So the singular vectors can be any
# sets of orthogonal vectors that span the space of the original matrix.
plt.plot(u[:,0])
plt.savefig('Figures/13.png', bbox_inches='tight')
plt.clf()

# We can see the same thing phase shifting the sine waves
data_shifted = np.transpose([np.sin((n*np.pi/20) + 
                             (np.array(range(0, 360, 2))*np.pi/180.))
                             for n in range(0, 21)])
plt.plot(data_shifted[:,0])
plt.plot(data_shifted[:,4])
plt.plot(data_shifted[:,8])
plt.savefig('Figures/14.png', bbox_inches='tight')
plt.clf()

# Taking the SVD
u, s, v = np.linalg.svd(data_scaled, full_matrices=0)
plt.scatter(range(len(s)), s)
plt.xticks(range(len(s)), [x + 1 for x in range(len(s))])
plt.savefig('Figures/15.png', bbox_inches='tight')
plt.clf()

# 4.
# Next we'll look at what happens if there are patterns in both dimensions. For
# example, this could arise in situations where a signal is measured over time.
# Start by scaling the amplitude of each column of the data by a different
# amount.
data_scaled = np.transpose([n * signal for n in range(1, 21)])
u, s, v = np.linalg.svd(data_scaled)
plt.scatter(range(len(s)), s)
plt.xticks(range(len(s)), [x + 1 for x in range(len(s))])
plt.savefig('Figures/16.png', bbox_inches='tight')
plt.clf()

# Here, again, all the information is captured in the first basis vector.
# Everything else is just a scaling of this.
plt.plot(u[:,0])
plt.savefig('Figures/17.png', bbox_inches='tight')
plt.clf()

# The scaling is visible in the right basis vector
plt.scatter(range(len(s)),v[0])
plt.savefig('Figures/18.png', bbox_inches='tight')
plt.clf()


# A second example is of sine waves with exponential decay over time.
time_signal = np.exp([-x for x in range(20)])
data_dynamic = np.outer(signal, time_signal)
plt.plot(data_dynamic[:,0])
plt.plot(data_dynamic[:,1])
plt.plot(data_dynamic[:,2])
plt.savefig('Figures/19.png', bbox_inches='tight')
plt.clf()

# Computing the SVD
u, s, v = np.linalg.svd(data_dynamic)
plt.scatter(range(len(s)), s)
plt.xticks(range(len(s)), [x + 1 for x in range(len(s))])
plt.savefig('Figures/20.png', bbox_inches='tight')
plt.clf()

# Here again the data is rank 1 so all the information is captured by the first
# left basis vector.
plt.plot(u[:,0])
plt.savefig('Figures/21.png', bbox_inches='tight')
plt.clf()

# And the corresponding right basis vector
plt.plot(v[0])
plt.savefig('Figures/22.png', bbox_inches='tight')
plt.clf()

# 5. Putting everything together, let's see what happens with multiple time-
# varying signals plus noise

# Generate a second signal that is quadratic. Scale it to similar size of the
# first signal so the results are easier to see.
signal_2 = [((x-180)/10)**2 for x in range(0, 360, 5)]
signal_2 = signal_2/np.max(signal_2)

# Generate a second time course. I decided to use a linear increase.
time_signal_2 = [x/20 for x in range(1, 21)]

# The second dataset is then the out product of the second signal and second
# time course.
data_dynamic_2 = np.outer(signal_2, time_signal_2)

# Sum the two datasets and the noise matrix
complicated_data = data_dynamic + data_dynamic_2 + noise

# Plot the first, fourth, and last columns to illustrate the signal.
plt.plot(complicated_data[:,0])
plt.plot(complicated_data[:,3])
plt.plot(complicated_data[:,19])
plt.savefig('Figures/23.png', bbox_inches='tight')
plt.clf()

# Compute SVD and view the singular values
u, s, v = np.linalg.svd(complicated_data)
plt.scatter(range(len(s)), s)
plt.xticks(range(len(s)), [x + 1 for x in range(len(s))])
plt.gca().set_yscale('log')
plt.savefig('Figures/24.png', bbox_inches='tight')
plt.clf()

# Plot the first two left basis vectors
plt.plot(u[:,0])
plt.plot(u[:,1])
plt.savefig('Figures/25.png', bbox_inches='tight')
plt.clf()

# Plot the first two right basis vectors
plt.plot(v[0])
plt.plot(v[1])
plt.savefig('Figures/26.png', bbox_inches='tight')
plt.clf()
