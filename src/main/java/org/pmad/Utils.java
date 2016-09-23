package org.pmad;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.regex.Pattern;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.apache.commons.math3.stat.correlation.Covariance;
import org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression;
import org.apache.commons.math3.util.MathArrays;
import org.apache.commons.math3.util.Pair;
import org.pmad.gmm.MyMND;
import org.pmad.gmm.MyMixMND;

import cc.mallet.pipe.CharSequence2TokenSequence;
import cc.mallet.pipe.CharSequenceLowercase;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.pipe.iterator.CsvIterator;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.util.Randoms;

public class Utils {

	static final String ROC = "roc";
	
	public static double[] lrSolve(double[][] a, double b []) {
		if (a.length != b.length) {
			System.err.println("not right!");
			return new double[0];
		}
		OLSMultipleLinearRegression lr = new OLSMultipleLinearRegression();
		lr.setNoIntercept(true);
		lr.newSampleData(b, a);
		return lr.estimateRegressionParameters();
	}
	
	public static double cosineSimilarity(double[] vectorA, double[] vectorB) {
	    double dotProduct = 0.0;
	    double normA = 0.0;
	    double normB = 0.0;
	    for (int i = 0; i < vectorA.length; i++) {
	        dotProduct += vectorA[i] * vectorB[i];
	        normA += Math.pow(vectorA[i], 2);
	        normB += Math.pow(vectorB[i], 2);
	    }   
	    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
	}
	public static double klDivergence(double[] p1, double[] p2) {
		double klDiv = 0.0;
		for (int i = 0; i < p1.length; ++i) {
			if (p1[i] == 0) {
				continue;
			}
			if (p2[i] == 0.0) {
				continue;
			}
			klDiv += p1[i] * Math.log(p1[i] / p2[i]);
		}
		return klDiv; // moved this division out of the loop -DM
	}

	public static double AP(int num, List<InsDists> resList) {
		double r = 0;
		double right = 0;
		double count = 0;
		for (int i = 0; i < resList.size(); i++) {
			count ++;
			if (resList.get(i).name.startsWith("a")) {
				right++;
				r += right/count;
				if (right == num) {
					break;
				}
			}		
		}
		return r/num;
	}
	
	public static double auc(List<InsDists> resList, String metName) {
		int totalPositive = 0;
		int totalNegative = 0;
		List<DoubleDoublePair> sortedProb = new ArrayList<DoubleDoublePair>();
		for (int i = 0; i < resList.size(); i++) {
			int label = resList.get(i).name.startsWith("a")?0:1;
			double key;
			if(metName.equals(Utils.ROC)) {
				key = resList.get(i).ensembleScore;
			} else {
				key = resList.get(i).distMap.get(metName);
			}
			sortedProb.add(new DoubleDoublePair(key, label));
			if (label == 0) {
				totalNegative++;
			} else {
				totalPositive++;
			}
		}
		Collections.sort(sortedProb);

		double fp = 0;
		double tp = 0;
		double fpPrev = 0;
		double tpPrev = 0;
		double area = 0;
		double fPrev = Double.MIN_VALUE;

		int i = 0;
		while (i < sortedProb.size()) {
			DoubleDoublePair pair = sortedProb.get(i);
			double curF = pair.key;
			if (curF != fPrev) {
				area += Math.abs(fp - fpPrev) * ((tp + tpPrev) / 2.0);
				fPrev = curF;
				fpPrev = fp;
				tpPrev = tp;
			}
			double label = pair.value;
			if (label == +1) {
				tp++;
			} else {
				fp++;
			}
			i++;
		}
		area += Math.abs(totalNegative - fpPrev) * ((totalPositive + tpPrev) / 2.0);
		area /= ((double) totalPositive * totalNegative);
		return area;
	}
	private static class DoubleDoublePair implements Comparable<DoubleDoublePair> {
		public double key;
		public double value;

		public DoubleDoublePair(double key, double value) {
			this.key = key;
			this.value = value;
		}

		@Override
		public int compareTo(DoubleDoublePair o) {
			if (this.key > o.key) {
				return -1;
			} else if (this.key < o.key) {
				return 1;
			}
			return 0;
		}
	}
	
	public static MyMixMND estimate(final double[][] data, final int numComponents)
			throws NotStrictlyPositiveException, DimensionMismatchException {
		if (data.length < 2) {
			throw new NotStrictlyPositiveException(data.length);
		}
		if (numComponents < 2) {
			throw new NumberIsTooSmallException(numComponents, 2, true);
		}
		if (numComponents > data.length) {
			throw new NumberIsTooLargeException(numComponents, data.length, true);
		}

		final int numRows = data.length;
		final int numCols = data[0].length;

		// sort the data
		final List<DataRow> sortedData = new ArrayList<>();
		for (int i = 0; i < numRows; i++) {
			sortedData.add(new DataRow(data[i]));
		}
		Collections.shuffle(sortedData, new Random(System.currentTimeMillis()));
		// uniform weight for each bin
		final double weight = 1d / numComponents;

		// components of mixture model to be created
		final List<Pair<Double, MyMND>> components = new ArrayList<Pair<Double, MyMND>>(
				numComponents);

		// create a component based on data in each bin
		for (int binIndex = 0; binIndex < numComponents; binIndex++) {
			// minimum index (inclusive) from sorted data for this bin
			final int minIndex = (binIndex * numRows) / numComponents;

			// maximum index (exclusive) from sorted data for this bin
			final int maxIndex = ((binIndex + 1) * numRows) / numComponents;

			// number of data records that will be in this bin
			final int numBinRows = maxIndex - minIndex;

			// data for this bin
			final double[][] binData = new double[numBinRows][numCols];

			// mean of each column for the data in the this bin
			final double[] columnMeans = new double[numCols];

			// populate bin and create component
			for (int i = minIndex, iBin = 0; i < maxIndex; i++, iBin++) {
				for (int j = 0; j < numCols; j++) {
					final double val = sortedData.get(i).getRow()[j];
					double val2 = new Random().nextDouble();
					columnMeans[j] += val2;
					binData[iBin][j] = val2;
				}
			}

			MathArrays.scaleInPlace(1d / numBinRows, columnMeans);

			// covariance matrix for this bin
			double[][] covMat = new Covariance(binData).getCovarianceMatrix().getData();
			MyMND mvn = null;
			try {
				mvn = new MyMND(columnMeans, covMat);
			} catch (SingularMatrixException e) {
//				System.out.println("add ridge...");
				//add a rigde to the main diagonal if singular
				for (int i = 0; i < covMat.length; i++) {
					covMat[i][i] += new Randoms().nextDouble()*0.1;
				}
				mvn = new MyMND(columnMeans, covMat);
			}

			components.add(new Pair<Double, MyMND>(weight, mvn));
		}

		return new MyMixMND(components);
	}
	
    /**
     * Class used for sorting user-supplied data.
     */
    private static class DataRow implements Comparable<DataRow> {
        /** One data row. */
        private final double[] row;
        /** Mean of the data row. */
        private Double mean;

        /**
         * Create a data row.
         * @param data Data to use for the row
         */
        DataRow(final double[] data) {
            // Store reference.
            row = data;
            // Compute mean.
            mean = 0d;
            for (int i = 0; i < data.length; i++) {
                mean += data[i];
            }
            mean /= data.length;
        }

        /**
         * Compare two data rows.
         * @param other The other row
         * @return int for sorting
         */
        public int compareTo(final DataRow other) {
            return mean.compareTo(other.mean);
        }

        /** {@inheritDoc} */
        @Override
        public boolean equals(Object other) {

            if (this == other) {
                return true;
            }

            if (other instanceof DataRow) {
                return MathArrays.equals(row, ((DataRow) other).row);
            }

            return false;

        }

        /** {@inheritDoc} */
        @Override
        public int hashCode() {
            return Arrays.hashCode(row);
        }
        /**
         * Get a data row.
         * @return data row array
         */
        public double[] getRow() {
            return row;
        }
    }

	public static double squareErr(double[] distB, double[] expected) {
		double r = 0;
		for (int i = 0; i < expected.length; i++) {
			double t = distB[i]-expected[i];
			r += t*t;
		}
		return r;
	}

	public static double innerProduct(double[] distA, double[] distB) {
		double r = 0;
		for (int i = 0; i < distA.length; i++) {
			r += distA[i]*distB[i];
		}
		return r;
	}

	public static double sum(List<Double> ensScores) {
		double sum = 0;
		for (Double double1 : ensScores) {
			sum += double1;
		}
		return sum;
	}

	public static boolean hasNaN(double[][] data) {
		for (int i = 0; i < data.length; i++) {
			double[] ds = data[i];
			for (int j = 0; j < ds.length; j++) {
				double d = ds[j];
				if (Double.isNaN(d)) {
					return true;
				}
			}
		}
		return false;
	}

	public static boolean hasZero(double[] gammaSums) {
		for (double d : gammaSums) {
			if (d == 0) {
				return true;
			}
		}
		return false;
	}

	public static void print(List<HashMap<String, Double>> rs) {
		if (rs.size() == 0) {
			return;
		}
		Set<String> keyset = rs.get(0).keySet();
		for (String string : keyset) {
			System.out.print(string + "\t");
		}
		System.out.println();
		for (HashMap<String, Double> r : rs) {
			for (String key : keyset) {
				System.out.print(r.get(key) + "\t");
			}
			System.out.println();
		}
	}

	public static void print(HashMap<String, Double> r, String fileName) throws IOException {
		Path path = Paths.get(fileName);
		BufferedWriter writer = null;
		if (!Files.exists(path)) {
			writer = Files.newBufferedWriter(path);
			Set<String> keyset = r.keySet();
			for (String string : keyset) {
				writer.write(string + "\t");
			}
			writer.newLine();
		}else {
			writer = Files.newBufferedWriter(path, StandardOpenOption.APPEND);
		}
		for (String key : r.keySet()) {
			writer.write(r.get(key) + "\t");
		}
		writer.newLine();
		writer.close();
	}
	
	public static void printPrintIntersect() throws IOException {
		List<String> lines = Files.readAllLines(Paths.get("D:\\zlxWorkplace\\anomaly\\poly\\real\\inter.csv"));
		Set<String> set20 = new HashSet<>();
		Set<String> set50 = new HashSet<>();
		Set<String> set100 = new HashSet<>();
		for (String line : lines) {
			String[] els = line.split(",");
			set20.add(els[0]);
			set50.add(els[1]);
			set100.add(els[2]);
		}
		set50.retainAll(set100);
		set50.stream().forEach(System.out::println);
	}

	public static void main(String[] args) throws IOException {
//		printPrintIntersect();
		double[] a = new double[]{1,2};
		double[] b = new double[]{-1,3};
		System.out.println(klDivergence(a, b));
	}

	public static double[][] BuildVector(InstanceList lansA) {
		int va = lansA.getAlphabet().size();
		double[][] res = new double[lansA.size()][va];
		for (int i = 0; i < lansA.size(); i++) {
			FeatureSequence tokens = (FeatureSequence) lansA.get(i).getData();
			for (int j = 0; j < tokens.size(); j++) {
				res[i][tokens.getIndexAtPosition(j)] ++;				
			}
		}
		return res;
	}

	public static InstanceList readInstances(String fileName) throws FileNotFoundException {
	
		List<Pipe> pipes = new ArrayList<>();
		pipes.add(new CharSequenceLowercase());
		pipes.add( new CharSequence2TokenSequence(Pattern.compile("\\S+")) );
		pipes.add( new TokenSequence2FeatureSequence() );
		
		InstanceList inses = new InstanceList(new SerialPipes(pipes));
		Reader reader = new InputStreamReader(new FileInputStream(new File(fileName)));
		inses.addThruPipe(new CsvIterator (reader, Pattern.compile("(\\S+)\\s(\\S+)\\s(.*)"),
	            3, 2, 1));
		
		return inses;
	}

}

