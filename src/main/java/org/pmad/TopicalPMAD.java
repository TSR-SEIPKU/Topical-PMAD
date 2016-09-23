package org.pmad;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

import cc.mallet.topics.PolylingualTopicModelMultiReadouts;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.InstanceList;


/**
 * @author PMAD
 * Main class for topical pmad
 * see the main method for detailed usage
 */
public class TopicalPMAD {

	public static final String FA_TRAIN = "data\\train-a";
	public static final String FB_TRAIN = "data\\train-b";
	public static final String FA_TEST = "data\\test-a";
	public static final String FB_TEST = "data\\test-b";
	public static final int NUM_ITER = 1000;
	
	public int numTopic;
	private InstanceList[] train = null;
	private PolylingualTopicModelMultiReadouts model = null;
	private SimilarityMeasures simMeasure = SimilarityMeasures.DOT;
	
	public TopicalPMAD(int numTopic) {
		this.numTopic = numTopic;
		train = new InstanceList[2];
	}


	public List<Double> getAnoScore(String faTest, String fbTest, int numIter) throws FileNotFoundException {
		List<Double> rs = new ArrayList<>();
		InstanceList insesAtest = Utils.readInstances(faTest);
		InstanceList insesBtest = Utils.readInstances(fbTest);
		TopicInferencer infA = model.getInferencer(0);
		TopicInferencer infB = model.getInferencer(1);
		for (int i = 0; i < insesAtest.size(); i++) {
			double[] distA = infA.getSampledDistribution(insesAtest.get(i), numIter, 10, 200);
			double[] distB = infB.getSampledDistribution(insesBtest.get(i), numIter, 10, 200);
			double as = this.getScore(distA, distB);
			rs.add(as);
		}
		return rs;
	}


	private double getScore(double[] distA, double[] distB) {
		switch (this.simMeasure) {
		case DOT:
			return -Utils.innerProduct(distA, distB);
		case EUC:
			return new EuclideanDistance().compute(distA, distB);
		case COS:
			return -Utils.cosineSimilarity(distA, distB);
		case PS:
			double rs = new PearsonsCorrelation().correlation(distA, distB);
			if (Double.isNaN(rs)) {
				rs = 0;
			}
			return -rs;
		case KL:
			return Utils.klDivergence(distA, distB);
		default:
			return -Utils.innerProduct(distA, distB);
		}
	}


	private void setSimMeasure(SimilarityMeasures m) {
		this.simMeasure  = m;
		
	}


	private void train(int numIter) {
		model = new PolylingualTopicModelMultiReadouts(numTopic, 50);
		model.addInstances(train);
		model.setNumIterations(numIter);
		model.showTopicsInterval = 1000;
		model.setOptimizeInterval(0);
		model.setBurninPeriod(200);
		model.estimate();
	}


	public void readTrainingData(String fa, String fb) throws FileNotFoundException {
		train[0] = Utils.readInstances(fa);
		train[1] = Utils.readInstances(fb);
	}
	
	

	public static void main(String[] args) throws FileNotFoundException {
		TopicalPMAD tPmad = new TopicalPMAD(20);
		//read training data from two files
		tPmad.readTrainingData(FA_TRAIN, FB_TRAIN);
		//train the model
		tPmad.train(NUM_ITER);
		//choose a similarity measure
		tPmad.setSimMeasure(SimilarityMeasures.DOT);
		//read and calculate the anomaly score for test data
		List<Double> anomalyScores = tPmad.getAnoScore(FA_TEST, FB_TEST, NUM_ITER);
		System.out.println(anomalyScores);
	}
}
