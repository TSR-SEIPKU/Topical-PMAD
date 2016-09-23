package org.pmad;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.ml.distance.CanberraDistance;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;

public class InsDists {

	public static final String[] DIST_METS = {"innerp", "cos", "ps", "euc", "kl"};
//	public static final String[] DIST_METS = {"kl"};
	public static final String[] NN_METS = {"innerp", "cos", "ps", "euc"};
	public static final String[] POINT_ANO_METS = {"svm", "lof", "gmm"};
	
	public List<Double> ensScores = new ArrayList<>();
	public InsDists(String name2, double[] distA, double[] distB) {
		name = name2;
		this.distMap.put("euc", -new EuclideanDistance().compute(distA, distB));
		this.distMap.put("innerp", Utils.innerProduct(distA, distB));
		this.distMap.put("cos", Utils.cosineSimilarity(distA, distB));
		this.distMap.put("ps", new PearsonsCorrelation().correlation(distA, distB));
		this.distMap.put("kl", - Utils.klDivergence(distA, distB) - Utils.klDivergence(distB, distA));
	}
	public InsDists(String name2, double[] distB, double[] expected, boolean isNN) {
		name = name2;
		this.distMap.put("euc", -new EuclideanDistance().compute(expected, distB));
		this.distMap.put("innerp", Utils.innerProduct(expected, distB));
		this.distMap.put("cos", Utils.cosineSimilarity(expected, distB));
		this.distMap.put("ps", new PearsonsCorrelation().correlation(expected, distB));
	}
	public InsDists(String name2) {
		name = name2;
	}
	public String name;
	
	public double ensembleScore = 0;
	public Map<String, Double> distMap = new HashMap<>();
	

	public static Comparator<InsDists> Enscomparator = (dist1, dist2) -> {
		double a1 = dist1.ensembleScore;
		double a2 = dist2.ensembleScore;
		if (a1 == a2) {
			return 0;
		} else if(a1 > a2) {
			return 1;
		} else return -1;
	};
	
	public static Comparator<? super InsDists> metComparator(String metName) {
		return (dist1, dist2) -> {
			double a1 = dist1.distMap.get(metName).isNaN()?Double.POSITIVE_INFINITY:dist1.distMap.get(metName);
			double a2 = dist2.distMap.get(metName).isNaN()?Double.POSITIVE_INFINITY:dist2.distMap.get(metName);
			if (a1 == a2) {
				return 0;
			} else if(a1 > a2) {
				return 1;
			} else return -1;
		};
	}
	
	public static void main(String[] args) {
		double a = Double.NaN;
		System.out.println(a);
	}
	public static double[] ensAPandROC(List<InsDists> distList, int numAno) {
		double[] r = new double[2];
		for (InsDists insDists : distList) {			
			insDists.ensembleScore = Utils.sum(insDists.ensScores);
		}
		Collections.sort(distList, InsDists.Enscomparator);
		r[0] = Utils.AP(numAno, distList);
		r[1] = Utils.auc(distList, Utils.ROC);
		return r;
	}

	public static void addEnsScore(List<InsDists> distList, String metName) {
		double lower = distList.get(0).distMap.get(metName);
		double upper = distList.get(distList.size()-1).distMap.get(metName);
		for (int i = 0; i < distList.size(); i++) {
//			double tmp = distList.get(i).distMap.get(metName);
//			distList.get(i).ensScores.add((tmp-lower)/(upper-lower));
			distList.get(i).ensScores.add((double) i);	
		}
	}
}
