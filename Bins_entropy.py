data['Binned'] = pd.qcut(data['Deviation'], q=5, labels=False)  # 0-4 labels



# Download Apple stock data
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")

# Compute EMA(50) and deviation
data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
data['Deviation'] = data['Close'] - data['EMA_50']

# Drop missing values (EMA requires 50 periods to start)
data.dropna(inplace=True)

# Reshape data for K-means (requires 2D array)
X = data['Deviation'].values.reshape(-1, 1)

# Use the Elbow Method to find optimal k
inertia = []
k_values = range(2, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Based on the Elbow Method, choose k (e.g., k=5)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
data['Bin'] = kmeans.fit_predict(X)


#Compute probabilities for each bin
probabilities = data['Bin'].value_counts(normalize=True, sort=False)
print("Probabilities:\n", probabilities)

# Calculate entropy
entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add epsilon to avoid log(0)
print(f"Entropy: {entropy:.2f} bits")
