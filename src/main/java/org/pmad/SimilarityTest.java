package org.pmad;

import java.util.List;

import org.apache.commons.math3.ml.distance.CanberraDistance;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;

import cc.mallet.types.Dirichlet;

public class SimilarityTest {

	public static void main(String[] args) {
		Dirichlet dir = new Dirichlet(20, 0.5);
		int n = 2000;
		System.out.println("euc\tcos\tspm\tps\tkl\tcan");
		while(n-- > 0) {
			double[] dis1 = dir.nextDistribution();
			double[] dis2 = dir.nextDistribution();
			System.out.print(new EuclideanDistance().compute(dis1, dis2) + "\t");
			System.out.print(Utils.cosineSimilarity(dis1, dis2) + "\t");
			System.out.print(new SpearmansCorrelation().correlation(dis1, dis2) + "\t");
			System.out.print(new PearsonsCorrelation().correlation(dis1, dis2) + "\t");
			System.out.print(Utils.klDivergence(dis1, dis2)+Utils.klDivergence(dis2, dis1) + "\t");
			System.out.println(new CanberraDistance().compute(dis1, dis2));
		}
	}

	public static void print(List<double[]> aS, List<double[]> bS) {
		System.out.println("euc,cos,ps,innerp");
		for (int i = 0; i < aS.size(); i++) {
			double[] dis1 = aS.get(i);
			double[] dis2 = bS.get(i);
			System.out.print(new EuclideanDistance().compute(dis1, dis2) + ",");
			System.out.print(Utils.cosineSimilarity(dis1, dis2) + ",");
			System.out.print(new PearsonsCorrelation().correlation(dis1, dis2) + ",");
			System.out.print(Utils.innerProduct(dis1, dis2) + "\n");
		}
	}
}
