

// This function calculates the covariance between two arrays using the Inner-Product Space Correction (ISPC) method.
export uniform double CovarianceISPC(const uniform double * uniform x, const uniform double * uniform y, const uniform int size)
{
	double a = 0., b = 0., c = 0.;
	foreach (i = 0 ... size)
	{
		a += x[i];
		b += y[i];
		c += x[i] * y[i];
	}
    uniform double ans = reduce_add(c) / size - (reduce_add(a) / size) * (reduce_add(b) / size);
	return ans;
}




