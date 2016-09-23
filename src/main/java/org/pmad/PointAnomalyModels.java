package org.pmad;


import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Vector;
import org.apache.commons.math3.linear.SingularMatrixException;
import org.pmad.gmm.MyMixMND;
import org.pmad.libsvm.svm;
import org.pmad.libsvm.svm_model;
import org.pmad.libsvm.svm_node;
import org.pmad.libsvm.svm_parameter;
import org.pmad.libsvm.svm_problem;

import weka.classifiers.misc.LOF;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class PointAnomalyModels {

	private static final int NUM_GMMCOM = 40;
	
	public static MyMixMND trainGMM(double[][] trainSet) {
//		int topicNum = trainSet[0].length;
//		MyMixMND gmm = null;
//		MyMixMNDEM gmmEM = null;
//		System.out.print("fitting Gaussian...");
//		gmm = Utils.estimate(trainSet, NUM_GMMCOM);
//		gmmEM = new MyMixMNDEM(trainSet);
//		gmmEM.fit(gmm);
//		System.out.println("done!");
//		gmm = gmmEM.getFittedModel();
//		return gmm;
		return null;
	}
	
	public static double testGMM(MyMixMND gmm, double[] dist) {
//		double[] distmut = new double[dist.length-1];
//		for (int j = 0; j < distmut.length; j++) {
//			distmut[j] = dist[j];
//		}
//		return gmm.density(dist);
		return 0;
	}
	
	public static void main(String[] args) throws Exception {
		 MyMixMND lof = trainGMM(new double[][]{
			{1,1,2,3},
			{1,2,2,1},
			{1,1,3,2},
			{1,2,1,3},
			{0,1,4,2},
			{0,0,3,4},
			{101,1,2,1},
			{101,2,2,2},
			{101,1,3,2},
			{101,2,1,3},
			{100,1,4,2},
			{100,0,3,4},
		});
		
		System.out.println(lof);
		System.out.println(testGMM(lof, new double[]{1,1,2.5,2.5}));
		System.out.println(testGMM(lof, new double[]{3,3,3,3}));
		System.out.println(testGMM(lof, new double[]{128,0.16,100.6,100.3}));
		System.out.println(testGMM(lof, new double[]{500,500,100,100}));
	}
	
	public static LOF trainLOF(double[][] trainSet) throws Exception {
		LOF lof = new LOF();
		String[] options = new String[6];
		options[0] = "-num-slots";
		options[1] = "4";
		options[2] = "-min";
		options[3] = "5";
		options[4] = "-max";
		options[5] = "10";
		lof.setOptions(options);
		
		int attriSize = trainSet[0].length;
		Instances train = getInsesForLOF(attriSize);
		for (int i = 0; i < trainSet.length; i++) {
			train.add(new DenseInstance(1, trainSet[i]));
		}
		List<String> bina = new ArrayList<>();
		bina.add("norm");bina.add("ab");
		train.insertAttributeAt(new Attribute("c", bina), attriSize);
		train.setClassIndex(attriSize);
		for (int i = 0; i < trainSet.length; i++) {
			train.get(i).setClassValue("ab");
		}
		System.out.print("building lof...");
		lof.buildClassifier(train);
		System.out.println("done!");
		
		return lof;
	}
	
	public static svm_model trainSVM(double[][] trainSet) throws Exception {
		svm_problem problem = new svm_problem();
		int dataCount = trainSet.length;
		int attriSize = trainSet[0].length;
		problem.y = new double[dataCount];
		problem.l = dataCount;
		problem.x = new svm_node[dataCount][];
		
		for (int i = 0; i < dataCount; i++) {
			problem.x[i] = new svm_node[attriSize];
			for (int j = 0; j < attriSize; j++) {
				svm_node node = new svm_node();
				node.index = j;
				node.value = trainSet[i][j];
				problem.x[i][j] = node;
			}
//			problem.y[i] = 0.2;
		}
		
		svm_parameter param = new svm_parameter();
		param.svm_type = svm_parameter.ONE_CLASS;
		param.kernel_type = svm_parameter.RBF;
		param.C = 1;
		param.gamma = 100;
		param.nu = 0.5;
		System.out.print("Trainning svm...");
		svm_model model = svm.svm_train(problem, param);
		System.out.println("Done!");
		return model;
	}

	public static Instances getInsesForLOF(int attriSize) {
		ArrayList<Attribute> attributes = new ArrayList<>();
		for (int i = 0; i < attriSize; i++) {
			attributes.add(new Attribute("a" + i));
		}	
		Instances train = new Instances("train", attributes, 0);
		return train;
	}

	public static Instances test = null;
	
	public static double testSVM(svm_model model, double[] dist) throws Exception {	
		int attriSize = dist.length;
		svm_node[] nodes = new svm_node[attriSize];
		for (int i = 0; i < attriSize; i++) {
			nodes[i] = new svm_node();
			nodes[i].index = i;
			nodes[i].value = dist[i];
		}
		double[] prob = new double[1];
		double r = svm.svm_predict_probability(model, nodes, prob);
		return r;
	}
	
	public static double testLOF(LOF lof, double[] dist) throws Exception {		
		int attriSize = dist.length;
		if (test == null) {
			test = getInsesForLOF(attriSize);
			List<String> bina = new ArrayList<>();
			bina.add("norm");bina.add("ab");
			test.insertAttributeAt(new Attribute("c", bina), attriSize);
			test.setClassIndex(attriSize);
		}
		test.add(new DenseInstance(1, dist));
//		test.get(0).setClassValue("ab");
		double r = lof.distributionForInstance(test.get(0))[0];
		test.remove(0);
		return r;
	}
	
}
