package org.pmad;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.commons.math3.exception.MaxCountExceededException;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;
import org.apache.commons.math3.util.Pair;
import org.pmad.gmm.MyMixMND;
import org.pmad.libsvm.svm;
import org.pmad.libsvm.svm_model;

import cc.mallet.topics.PolylingualTopicModelMultiReadouts;
import cc.mallet.topics.PolylingualTopicModelMultiReadouts.TopicAssignment;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import weka.classifiers.misc.LOF;

/**
 * @author PMAD
 * The class for cross validations on various dataset with PMAD and all the baseline method
 * See the Main method
 */
public class Experiments {

	private static final String DIA_MED_DISTS_TXT = "-dia-med-dists.txt";
	public static final String SUB_DIR_R = "real";
	public static final String SUB_DIR_S = "synthetic";
	
	/*
	 * Working directory
	 */
	public static final String DIR = "data" + SUB_DIR_R + "\\";
	
	
	public static final int MED = 0;
	public static final int DIA = 1;
	
	/*
	 * number of normal/anomalous instances
	 */
//	public static final int NUM_NORM = 9900;
//	public static final int NUM_AB = 100;
	public static final int NUM_NORM = 46826;
	public static final int NUM_AB = 477;
	
	public static final Set<String> TOP_LIST = new HashSet<>(Arrays.asList(
			));
	
	public static void main(String[] args) throws Exception {


		int numCV = 5;//number of folds
		int topicNum = 50;
		int timeRun = 10; //times of run
		String fa = "med-sh";//file name of input data - Type-A features
		String fb = "dia-sh";//file name of input data - Type-B features
		
		
		for (int i = 0; i < timeRun; i++) {
			// shuffle instances
			InstanceList instencesA = readAndShuffle(fa);
			InstanceList instencesB = readAndSameShuffle(instencesA, fb);
			System.out.println(fa + "------------------" + i + "------------------------" + fb);
			
			//print result to files
			
			//results of topical PMAD
			Utils.print(CrossValPMAD(instencesA, instencesB, numCV, topicNum),DIR+fa+topicNum+"PMAD");
			//results of contextual anomaly detection
			Utils.print(CrossValCAD(instencesA, instencesB, numCV, topicNum),DIR+fa+topicNum+"CAD");
			//results of point anomaly detection
			Utils.print(CrossValPAD(instencesA, instencesB, numCV, topicNum),DIR+fa+topicNum+"PAD");
		}
		

//		String modelName = "ori-50";
//		EstimateNewModel(new String[]{
//				"med",
//				"dia"
//		}, modelName, 50);
//		PolylingualTopicModelMultiReadouts pltm = PolylingualTopicModelMultiReadouts.read(new File(DIR + modelName));
//		calculateAndWriteDist(modelName, pltm);
//		readDistrAndWriteDis(modelName, pltm);
//		writeDocDetails(modelName, pltm);
//		System.out.println(pltm.modelLogLikelihood());
		
	}

	
//	private static HashMap<String, Double> CrossValPCA(InstanceList instencesA, InstanceList instencesB, int nCV) {
//		HashMap<String, Double> resultsAP = new HashMap<>();
////		HashMap<String, Double> resultsROC = new HashMap<>();
//		for (int distm = 0; distm < InsDists.NN_METS.length; distm++) {
//			resultsAP.put(InsDists.NN_METS[distm], .0);
//		}
//		for (int cv = 0; cv < nCV; cv++) {
//			System.out.println("CV" + cv);
//			InstanceList lansA = new InstanceList(instencesA.getPipe());
//			lansA.addAll(instencesA.subList(0, cv*NUM_NORM/nCV));
//			lansA.addAll(instencesA.subList((cv+1)*NUM_NORM/nCV, nCV*NUM_NORM/nCV));
//			lansA.addAll(instencesA.subList(NUM_NORM, NUM_NORM + cv*NUM_AB/nCV));
//			lansA.addAll(instencesA.subList(NUM_NORM + (cv+1)*NUM_AB/nCV, NUM_NORM + nCV*NUM_AB/nCV));
//			InstanceList lansB = new InstanceList(instencesB.getPipe());
//			lansB.addAll(instencesB.subList(0, cv*NUM_NORM/nCV));
//			lansB.addAll(instencesB.subList((cv+1)*NUM_NORM/nCV, nCV*NUM_NORM/nCV));
//			lansB.addAll(instencesB.subList(NUM_NORM, NUM_NORM + cv*NUM_AB/nCV));
//			lansB.addAll(instencesB.subList(NUM_NORM + (cv+1)*NUM_AB/nCV, NUM_NORM + nCV*NUM_AB/nCV));
//			
//			double[][] testMatrixA = Utils.BuildVector(lansA);
//			double[][] testMatrixB = Utils.BuildVector(lansB);
//			
//			
//			
//			MultiLayerPerceptron mlp2 = new MultiLayerPerceptron(testMatrixA, testMatrixB, true);
//
//			InstanceList insesAtest = instencesA.subList(cv*NUM_NORM/nCV, (cv+1)*NUM_NORM/nCV);
//			insesAtest.addAll(instencesA.subList(NUM_NORM + cv*NUM_AB/nCV, NUM_NORM + (cv+1)*NUM_AB/nCV));
//			double[][] testAM = Utils.BuildVector(insesAtest);
//			InstanceList insesBtest = instencesB.subList(cv*NUM_NORM/nCV, (cv+1)*NUM_NORM/nCV);
//			insesBtest.addAll(instencesB.subList(NUM_NORM + cv*NUM_AB/nCV, NUM_NORM + (cv+1)*NUM_AB/nCV));
//			double[][] testBM = Utils.BuildVector(insesBtest);
//			List<InsDists> distList = new ArrayList<>();
//			for (int i = 0; i < insesAtest.size(); i++) {
//				String name = (String) insesBtest.get(i).getName();
//				double[] distA = testAM[i];
//				double[] distB = testBM[i];
//				double[] expected = mlp2.fit(distA);
//				InsDists dists = new InsDists(name, distB, expected, true);
//				distList.add(dists);				
//			}
//			for (int distm = 0; distm < InsDists.NN_METS.length; distm++) {
//				String metName = InsDists.NN_METS[distm];
//				Collections.sort(distList, InsDists.metComparator(metName));
////				for (int i = 0; i < distList.size(); i++) {
////					distList.get(i).ensembleScore += i;
////				}
//				resultsAP.put(metName, resultsAP.get(metName) + Utils.AP(NUM_AB/nCV, distList)/nCV);
////				resultsROC.put(metName, resultsROC.get(metName) + Utils.auc(distList, metName));
//			}
////			Collections.sort(distList, InsDists.Enscomparator);
////			ensRes += Utils.AP(NUM_AB/nCV, distList);
//			System.out.println("AP:  " + resultsAP);
////			System.out.println("ROC: " + resultsROC);
////			System.out.println(ensRes);
//		}
//		System.out.println("AP:  ");
//		for (String metName : InsDists.NN_METS) {
//			System.out.println(metName + "\t" + resultsAP.get(metName));
//		}
////		System.out.println("ROC:  ");
////		for (String metName : InsDists.NN_METS) {
////			System.out.println(metName + "\t" + resultsROC.get(metName)/nCV);
////		}
////		System.out.println(ensRes/nCV);	
//		return resultsAP;
//	}


	private static HashMap<String, Double> CrossValPAD(InstanceList instencesA, InstanceList instencesB, int nCV, int topicNum) throws Exception {
		HashMap<String, Double> resultsAP = new HashMap<>();
//		HashMap<String, Double> resultsROC = new HashMap<>();
		for (int distm = 0; distm < InsDists.POINT_ANO_METS.length; distm++) {
			resultsAP.put(InsDists.POINT_ANO_METS[distm], .0);
//			resultsROC.put(InsDists.POINT_ANO_METS[distm], .0);
		}
		int errCount = 0;
		for (int cv = 0; cv < nCV; cv++) {
			System.out.println("CV" + cv);
			InstanceList[] lansA = new InstanceList[1];
			InstanceList[] lansB = new InstanceList[1];
			lansA[0] = new InstanceList(instencesA.getPipe());
			lansA[0].addAll(instencesA.subList(0, cv*NUM_NORM/nCV));
			lansA[0].addAll(instencesA.subList((cv+1)*NUM_NORM/nCV, nCV*NUM_NORM/nCV));
			lansA[0].addAll(instencesA.subList(NUM_NORM, NUM_NORM + cv*NUM_AB/nCV));
			lansA[0].addAll(instencesA.subList(NUM_NORM + (cv+1)*NUM_AB/nCV, NUM_NORM + nCV*NUM_AB/nCV));
			lansB[0] = new InstanceList(instencesB.getPipe());
			lansB[0].addAll(instencesB.subList(0, cv*NUM_NORM/nCV));
			lansB[0].addAll(instencesB.subList((cv+1)*NUM_NORM/nCV, nCV*NUM_NORM/nCV));
			lansB[0].addAll(instencesB.subList(NUM_NORM, NUM_NORM + cv*NUM_AB/nCV));
			lansB[0].addAll(instencesB.subList(NUM_NORM + (cv+1)*NUM_AB/nCV, NUM_NORM + nCV*NUM_AB/nCV));
			PolylingualTopicModelMultiReadouts pltmA = new PolylingualTopicModelMultiReadouts(topicNum, 50);
			PolylingualTopicModelMultiReadouts pltmB = new PolylingualTopicModelMultiReadouts(topicNum, 50);
			pltmA.addInstances(lansA);
			pltmA.setNumIterations(1000);
			pltmA.showTopicsInterval = 1000;
			pltmA.setOptimizeInterval(0);
			pltmA.setBurninPeriod(200);
			pltmA.estimate();
			pltmB.addInstances(lansB);
			pltmB.setNumIterations(1000);
			pltmB.showTopicsInterval = 1000;
			pltmB.setOptimizeInterval(0);
			pltmB.setBurninPeriod(200);
			pltmB.estimate();
			
			double[][] testMatrixA = pltmA.getDocTops();
			double[][] testMatrixB = pltmB.getDocTops();
			
			int numIns = testMatrixA.length;
			double[][] trainSet = new double[numIns][topicNum*2-2];
			for (int i = 0; i < trainSet.length; i++) {
				double[] ds = trainSet[i];
				for (int j = 0; j < topicNum-1; j++) {
					ds[j] = testMatrixA[i][j];
				}
				for (int j = topicNum-1; j < topicNum*2-2; j++) {
					ds[j] = testMatrixB[i][j-topicNum+1];
				}
			}
			
			MyMixMND gmm;
			try {
				gmm = PointAnomalyModels.trainGMM(trainSet);
				errCount = 0;
			} catch (MaxCountExceededException e) {
				System.err.println("ERR" + errCount);
				errCount ++;
				if (errCount < 3) {
					cv--;
				}
				continue;
			}
			LOF lof = PointAnomalyModels.trainLOF(trainSet);
			svm_model svm = PointAnomalyModels.trainSVM(trainSet);

			InstanceList insesAtest = instencesA.subList(cv*NUM_NORM/nCV, (cv+1)*NUM_NORM/nCV);
			insesAtest.addAll(instencesA.subList(NUM_NORM + cv*NUM_AB/nCV, NUM_NORM + (cv+1)*NUM_AB/nCV));
			InstanceList insesBtest = instencesB.subList(cv*NUM_NORM/nCV, (cv+1)*NUM_NORM/nCV);
			insesBtest.addAll(instencesB.subList(NUM_NORM + cv*NUM_AB/nCV, NUM_NORM + (cv+1)*NUM_AB/nCV));
			TopicInferencer infA = pltmA.getInferencer(0);
			TopicInferencer infB = pltmB.getInferencer(0);
			List<InsDists> resList = new ArrayList<>();
			for (int i = 0; i < insesAtest.size(); i++) {
				String name = (String) insesBtest.get(i).getName();
				double[] distA = infA.getSampledDistribution(insesAtest.get(i), 1000, 10, 200);
				double[] distB = infB.getSampledDistribution(insesBtest.get(i), 1000, 10, 200);
				double[] dist = new double[topicNum*2-2];
				for (int j = 0; j < topicNum-1; j++) {
					dist[j] = distA[j];
				}
				for (int j = topicNum-1; j < topicNum*2-2; j++) {
					dist[j] = distB[j-topicNum+1];
				}
				InsDists pa = new InsDists(name);
				
				pa.distMap.put("gmm", PointAnomalyModels.testGMM(gmm, dist));
				pa.distMap.put("lof", PointAnomalyModels.testLOF(lof, dist));
				pa.distMap.put("svm", PointAnomalyModels.testSVM(svm, dist));
//				System.out.println(pa.distMap);
				resList.add(pa);
			}
			for (int distm = 0; distm < InsDists.POINT_ANO_METS.length; distm++) {
				String metName = InsDists.POINT_ANO_METS[distm];
				Collections.sort(resList, InsDists.metComparator(metName));
				resultsAP.put(metName, resultsAP.get(metName) + Utils.AP(NUM_AB/nCV, resList)/nCV);
//				resultsROC.put(metName, resultsROC.get(metName) + Utils.auc(resList, metName));
//				System.out.println(ensRes);
			}
			System.out.println("AP:  " + resultsAP);
//			System.out.println("ROC: " + resultsROC);
		}
		System.out.println("AP:  ");
		for (String metName : InsDists.POINT_ANO_METS) {
			System.out.println(metName + "\t" + resultsAP.get(metName));
		}
//		System.out.println("ROC:  ");
//		for (String metName : InsDists.POINT_ANO_METS) {
//			System.out.println(metName + "\t" + resultsROC.get(metName)/nCV);
//		}
		return resultsAP;
	}

	

	private static HashMap<String, Double> CrossValCAD(InstanceList instencesA, InstanceList instencesB, int nCV, int topicNum) throws IOException {
		HashMap<String, Double> resultsAP = new HashMap<>();
//		HashMap<String, Double> resultsROC = new HashMap<>();
		for (int distm = 0; distm < InsDists.NN_METS.length; distm++) {
			resultsAP.put(InsDists.NN_METS[distm], .0);
//			resultsROC.put(InsDists.NN_METS[distm], .0);
		}
		for (int cv = 0; cv < nCV; cv++) {
			InstanceList[] lansA = new InstanceList[1];
			InstanceList[] lansB = new InstanceList[1];
			lansA[0] = new InstanceList(instencesA.getPipe());
			lansA[0].addAll(instencesA.subList(0, cv*NUM_NORM/nCV));
			lansA[0].addAll(instencesA.subList((cv+1)*NUM_NORM/nCV, nCV*NUM_NORM/nCV));
			lansA[0].addAll(instencesA.subList(NUM_NORM, NUM_NORM + cv*NUM_AB/nCV));
			lansA[0].addAll(instencesA.subList(NUM_NORM + (cv+1)*NUM_AB/nCV, NUM_NORM + nCV*NUM_AB/nCV));
			lansB[0] = new InstanceList(instencesB.getPipe());
			lansB[0].addAll(instencesB.subList(0, cv*NUM_NORM/nCV));
			lansB[0].addAll(instencesB.subList((cv+1)*NUM_NORM/nCV, nCV*NUM_NORM/nCV));
			lansB[0].addAll(instencesB.subList(NUM_NORM, NUM_NORM + cv*NUM_AB/nCV));
			lansB[0].addAll(instencesB.subList(NUM_NORM + (cv+1)*NUM_AB/nCV, NUM_NORM + nCV*NUM_AB/nCV));
			PolylingualTopicModelMultiReadouts pltmA = new PolylingualTopicModelMultiReadouts(topicNum, 50);
			PolylingualTopicModelMultiReadouts pltmB = new PolylingualTopicModelMultiReadouts(topicNum, 50);
			pltmA.addInstances(lansA);
			pltmA.setNumIterations(1000);
			pltmA.showTopicsInterval = 1000;
			pltmA.setOptimizeInterval(0);
			pltmA.setBurninPeriod(200);
			pltmA.estimate();
			pltmB.addInstances(lansB);
			pltmB.setNumIterations(1000);
			pltmB.showTopicsInterval = 1000;
			pltmB.setOptimizeInterval(0);
			pltmB.setBurninPeriod(200);
			pltmB.estimate();
			
			double[][] testMatrixA = pltmA.getDocTops();
			double[][] testMatrixB = pltmB.getDocTops();
//			M2MLR model = new M2MLR(testMatrixA, testMatrixB);
			MultiLayerPerceptron model = new MultiLayerPerceptron(testMatrixA, testMatrixB);

			InstanceList insesAtest = instencesA.subList(cv*NUM_NORM/nCV, (cv+1)*NUM_NORM/nCV);
			insesAtest.addAll(instencesA.subList(NUM_NORM + cv*NUM_AB/nCV, NUM_NORM + (cv+1)*NUM_AB/nCV));
			InstanceList insesBtest = instencesB.subList(cv*NUM_NORM/nCV, (cv+1)*NUM_NORM/nCV);
			insesBtest.addAll(instencesB.subList(NUM_NORM + cv*NUM_AB/nCV, NUM_NORM + (cv+1)*NUM_AB/nCV));
			TopicInferencer infA = pltmA.getInferencer(0);
			TopicInferencer infB = pltmB.getInferencer(0);
			List<InsDists> distList = new ArrayList<>();
			for (int i = 0; i < insesAtest.size(); i++) {
				String name = (String) insesBtest.get(i).getName();
				double[] distA = infA.getSampledDistribution(insesAtest.get(i), 1000, 10, 200);
				double[] distB = infB.getSampledDistribution(insesBtest.get(i), 1000, 10, 200);
				double[] expected = model.fit(distA);
				InsDists dists = new InsDists(name, distB, expected, true);
				distList.add(dists);				
			}
			for (int distm = 0; distm < InsDists.NN_METS.length; distm++) {
				String metName = InsDists.NN_METS[distm];
				Collections.sort(distList, InsDists.metComparator(metName));
//				for (int i = 0; i < distList.size(); i++) {
//					distList.get(i).ensembleScore += i;
//				}
				resultsAP.put(metName, resultsAP.get(metName) + Utils.AP(NUM_AB/nCV, distList)/nCV);
//				resultsROC.put(metName, resultsROC.get(metName) + Utils.auc(distList, metName));
			}
//			Collections.sort(distList, InsDists.Enscomparator);
//			ensRes += Utils.AP(NUM_AB/nCV, distList);
			System.out.println("AP:  " + resultsAP);
//			System.out.println("ROC: " + resultsROC);
//			System.out.println(ensRes);
		}
		System.out.println("AP:  ");
		for (String metName : InsDists.NN_METS) {
			System.out.println(metName + "\t" + resultsAP.get(metName));
		}
//		System.out.println("ROC:  ");
//		for (String metName : InsDists.NN_METS) {
//			System.out.println(metName + "\t" + resultsROC.get(metName)/nCV);
//		}
//		System.out.println(ensRes/nCV);	
		return resultsAP;
	}




	private static HashMap<String, Double> CrossValPMAD(InstanceList instencesA, InstanceList instencesB, int nCV, int topicNum) throws IOException {
		
		HashMap<String, Double> resultsAP = new HashMap<>();
//		HashMap<String, Double> resultsROC = new HashMap<>();
//		List<double[]> AS = new ArrayList<>();
//		List<double[]> BS = new ArrayList<>();
		for (int distm = 0; distm < InsDists.DIST_METS.length; distm++) {
			resultsAP.put(InsDists.DIST_METS[distm], .0);
//			resultsROC.put(InsDists.DIST_METS[distm], .0);
		}
//		double ensROC = 0;
//		double ensAP = 0;
		for (int cv = 0; cv < nCV; cv++) {
			InstanceList[] lans = new InstanceList[2];
			lans[0] = new InstanceList(instencesA.getPipe());
			lans[0].addAll(instencesA.subList(0, cv*NUM_NORM/nCV));
			lans[0].addAll(instencesA.subList((cv+1)*NUM_NORM/nCV, nCV*NUM_NORM/nCV));
			lans[0].addAll(instencesA.subList(NUM_NORM, NUM_NORM + cv*NUM_AB/nCV));
			lans[0].addAll(instencesA.subList(NUM_NORM + (cv+1)*NUM_AB/nCV, NUM_NORM + nCV*NUM_AB/nCV));
			lans[1] = new InstanceList(instencesB.getPipe());
			lans[1].addAll(instencesB.subList(0, cv*NUM_NORM/nCV));
			lans[1].addAll(instencesB.subList((cv+1)*NUM_NORM/nCV, nCV*NUM_NORM/nCV));
			lans[1].addAll(instencesB.subList(NUM_NORM, NUM_NORM + cv*NUM_AB/nCV));
			lans[1].addAll(instencesB.subList(NUM_NORM + (cv+1)*NUM_AB/nCV, NUM_NORM + nCV*NUM_AB/nCV));
			PolylingualTopicModelMultiReadouts pltm = new PolylingualTopicModelMultiReadouts(topicNum, 50);
			pltm.addInstances(lans);
			pltm.setNumIterations(1000);
			pltm.showTopicsInterval = 1000;
			pltm.setOptimizeInterval(0);
			pltm.setBurninPeriod(200);
			pltm.estimate(); 

			InstanceList insesAtest = instencesA.subList(cv*NUM_NORM/nCV, (cv+1)*NUM_NORM/nCV);
			insesAtest.addAll(instencesA.subList(NUM_NORM + cv*NUM_AB/nCV, NUM_NORM + (cv+1)*NUM_AB/nCV));
			InstanceList insesBtest = instencesB.subList(cv*NUM_NORM/nCV, (cv+1)*NUM_NORM/nCV);
			insesBtest.addAll(instencesB.subList(NUM_NORM + cv*NUM_AB/nCV, NUM_NORM + (cv+1)*NUM_AB/nCV));
			TopicInferencer infA = pltm.getInferencer(0);
			TopicInferencer infB = pltm.getInferencer(1);
			List<InsDists> distList = new ArrayList<>();
			for (int i = 0; i < insesAtest.size(); i++) {
				String name = (String) insesBtest.get(i).getName();
				double[] distA = infA.getSampledDistribution(insesAtest.get(i), 1000, 10, 200);
				double[] distB = infB.getSampledDistribution(insesBtest.get(i), 1000, 10, 200);
//				if(i < 200) {
//					AS.add(distA);
//					BS.add(distB);
//				}
				InsDists dists = new InsDists(name, distA, distB);
				distList.add(dists);
			}
			for (int distm = 0; distm < InsDists.DIST_METS.length; distm++) {
				String metName = InsDists.DIST_METS[distm];
				Collections.sort(distList, InsDists.metComparator(metName));
//				InsDists.addEnsScore(distList, metName);
				resultsAP.put(metName, resultsAP.get(metName) + Utils.AP(NUM_AB/nCV, distList)/nCV);
//				resultsROC.put(metName, resultsROC.get(metName) + Utils.auc(distList, metName));
			}
//			double[] ensAPandROC = InsDists.ensAPandROC(distList, NUM_AB/nCV);
			
//			ensAP += ensAPandROC[0];
//			ensROC += ensAPandROC[1];
			System.out.println("AP:  " + resultsAP);
//			System.out.println("ROC: " + resultsROC);
		}
		System.out.println("AP:  ");
//		System.out.println(ensAP/nCV);
		for (String metName : InsDists.DIST_METS) {
			System.out.println(metName + "\t" + resultsAP.get(metName));
		}
//		System.out.println("ROC:  ");
//		System.out.println(ensROC/nCV);
//		for (String metName : InsDists.DIST_METS) {
//			System.out.println(metName + "\t" + resultsROC.get(metName)/nCV);
//		}
//		SimilarityTest.print(AS,BS);
		return resultsAP;
	}

	private static InstanceList readAndSameShuffle(InstanceList instencesA, String fileB) throws FileNotFoundException {
		InstanceList instencesB = Utils.readInstances(Experiments.DIR + fileB);
		InstanceList instencesBShuffled = new InstanceList(instencesB.getPipe(), NUM_AB+NUM_NORM);
		instencesBShuffled.addAll(instencesB);
		int index = 0;
		for (Instance instance : instencesA) {
			String name = (String) instance.getName();
			if (name.startsWith("a")) {
				int iA = Integer.parseInt(name.substring(3));
				instencesBShuffled.set(index, instencesB.get(iA + NUM_NORM));
			} else {
				int iA = Integer.parseInt(name);
				instencesBShuffled.set(index, instencesB.get(iA));
			}
			index ++;
		}
		return instencesBShuffled;
	}

	public static InstanceList readAndShuffle(String fileA) throws FileNotFoundException {
		InstanceList instencesA = Utils.readInstances(Experiments.DIR + fileA);
		InstanceList normA = instencesA.subList(0, NUM_NORM);
		InstanceList abnormA = instencesA.subList(NUM_NORM, NUM_NORM+NUM_AB);
		normA.shuffle(new Random());
		abnormA.shuffle(new Random());
		InstanceList instencesAShuffled = new InstanceList(instencesA.getPipe());
		instencesAShuffled.addAll(normA);
		instencesAShuffled.addAll(abnormA);
		return instencesAShuffled;
	}




	private static void writeDocDetails(String modelName, PolylingualTopicModelMultiReadouts pltm) throws IOException {
			
		double[][][] topword = pltm.getTopWordsDist();
		double[][] topMed = topword[MED];
		double[][] topDia = topword[DIA];
		ArrayList<TopicAssignment> Ins = pltm.getData();
		
		List<String> distStrings = Files.readAllLines(Paths.get(DIR + modelName + DIA_MED_DISTS_TXT));

		int i = 0;
		int index = 0;
		BufferedWriter writer = Files.newBufferedWriter(Paths.get(DIR + modelName + ".docdetail"));
		while (i < distStrings.size()) {
			String name = distStrings.get(i);
			if (!TOP_LIST.contains(name)) {
				i += pltm.getNumTopics()+1;
				index++;
				continue;
			}
//			rewriter.write(name + "\n");
			Pair<Integer, Double>[] distDia = new Pair[pltm.getNumTopics()];
			Pair<Integer, Double>[] distMed = new Pair[pltm.getNumTopics()];
			for (int j = 0; j < pltm.getNumTopics(); j++) {
				String[] diString = distStrings.get(++i).split("\t");
//				rewriter.write(j + "\t" + distStrings.get(i) + "\n");
				Pair<Integer, Double> diapair = new Pair<Integer, Double>(j, Double.parseDouble(diString[1]));
				distDia[j] = diapair;
				Pair<Integer, Double> medpair = new Pair<Integer, Double>(j, Double.parseDouble(diString[2]));
				distMed[j] = medpair;
			}
			i++;
			writer.write(name + "\n");
//			writer.write(Ins.get(i).instances[MED].getData().toString() + "\n");
			FeatureSequence tokens = (FeatureSequence) Ins.get(index).instances[DIA].getData();
			for (int j = 0; j < tokens.size(); j++) {
//				List<String> topProbList = new ArrayList<>();
				int maxK = -1;
				double maxProb = 0; 
				for (int k = 0; k < pltm.getNumTopics(); k++) {
					double prob = distDia[k].getValue() * topDia[k][tokens.getIndexAtPosition(j)];
					if (prob > maxProb) {
						maxProb = prob;
						maxK = k;
					}
				}
				writer.write(tokens.get(j) + ":" + maxK+" ");
			}
			writer.newLine();
			Arrays.sort(distDia, (p1,p2) -> {
				if (p1.getValue() > p2.getValue()) {
					return -1;
				} else if (p1.getValue() < p2.getValue()) {
					return 1;
				} else {
					return 0;
				}
			});
			writer.write("Dia:");
			for (int j = 0; j < 5; j++) {
				writer.write(distDia[j].getKey() + ":" + distDia[j].getValue() + ", ");
			}
			writer.newLine();
			tokens = (FeatureSequence) Ins.get(index).instances[MED].getData();
			Map<Integer, Integer> medcount = new HashMap<>();
			for (int j = 0; j < tokens.size(); j++) {
				int medIndex = tokens.getIndexAtPosition(j);
				medcount.put(medIndex, medcount.getOrDefault(medIndex, 0) + 1);
			}
			Set<Integer> medSet = new HashSet<>();
			for (int j = 0; j < tokens.size(); j++) {
				int medIndex = tokens.getIndexAtPosition(j);
				if (medSet.contains(medIndex)) {
					continue;
				}
				medSet.add(medIndex);
				int maxK = -1;
				double maxProb = 0;
				List<String> probs = new ArrayList<>();
				for (int k = 0; k < pltm.getNumTopics(); k++) {
					double prob = distMed[k].getValue() * topMed[k][medIndex];
					probs.add(k + "|" + prob);
					if (prob > maxProb) {
						maxProb = prob;
						maxK = k;
					}
				}
				Collections.sort(probs, (s1,s2) -> {
					double d1 = Double.parseDouble(s1.split("\\|")[1]);
					double d2 = Double.parseDouble(s2.split("\\|")[1]);
					if (d1 > d2) {
						return -1;
					} else if (d1 < d2) {
						return 1;
					} else {
						return 0;
					}
				});
				writer.write(tokens.get(j) + "*" + medcount.get(medIndex)+"\n");
				writer.write(probs.subList(0, 5).toString() + "\n");
			}
			Arrays.sort(distMed, (p1,p2) -> {
				if (p1.getValue() > p2.getValue()) {
					return -1;
				} else if (p1.getValue() < p2.getValue()) {
					return 1;
				} else {
					return 0;
				}
			});
			writer.write("Med:");
			for (int j = 0; j < 5; j++) {
				writer.write(distMed[j].getKey() + ":" + distMed[j].getValue() + ", ");
			}
			writer.newLine();
			writer.newLine();
			index ++;
			if (index%1000 == 0) {
				System.out.println(index);
			}
		}

		
		writer.close();
	}



	public static void calculateAndWriteDist(String modelName, PolylingualTopicModelMultiReadouts pltm)
			throws IOException {
		
		TopicInferencer infdia = pltm.getInferencer(DIA);
		TopicInferencer infmed = pltm.getInferencer(MED);
		ArrayList<TopicAssignment> datapltm = pltm.getData();
		int m = 0;
		BufferedWriter differwriter = Files.newBufferedWriter(Paths.get(DIR + modelName + DIA_MED_DISTS_TXT));
		for (TopicAssignment ta : datapltm) {
			Instance inseMed = ta.instances[MED];
			Instance inseDia = ta.instances[DIA];
			differwriter.write(inseMed.getName() + "\n");
			double[] distDia = infdia.getSampledDistribution(inseDia, 1000, 10, 200);
			double[] distMed = infmed.getSampledDistribution(inseMed, 1000, 10, 200);
			for (int i = 0; i < distMed.length; i++) {
				differwriter.write(i + "\t" + distDia[i] + "\t" + distMed[i] + "\n");				
			}
			m++;
			if(m%1000 == 0) System.out.println(m);
		}
		differwriter.close();
	}

	//===============calculate med log=================
//	double[] ndw = new double[pltm.vocabularySizes[MED]];
//	FeatureSequence tokens = (FeatureSequence) inseMed.getData();
//	double logl = 0;
//	for (int position = 0; position < tokens.getLength(); position++) {
//		int word = tokens.getIndexAtPosition(position);
//		ndw[word]++;
//	}
//	for (int w = 0; w < ndw.length; w++){ 
//		double r = ((double)ndw[w])/tokens.getLength();
//		double p = 0;
//		for(int k = 0; k < pltm.getNumTopics(); k++) {
//			p += dist[k] * topMed[k][w];
//		}
//		p = p / medNormFactor.get(dict.lookupObject(w));
//		double logp = Math.log(p);
//		logl += r*logp;
//	}
	
	private static void readDistrAndWriteDis(String modelName, PolylingualTopicModelMultiReadouts pltm) throws Exception {
		List<String> distStrings = Files.readAllLines(Paths.get(DIR + modelName + DIA_MED_DISTS_TXT));
		BufferedWriter writer = Files.newBufferedWriter(Paths.get(DIR + modelName + "-polyresult-met.csv"));
//		BufferedWriter rewriter = Files.newBufferedWriter(Paths.get(DIR + modelName + "---DIA_MED_DISTS_TXT"));
		int t = 1;
		while (distStrings.get(t).contains("\t")) {
			t++;
		}
		t--;
		System.out.println(t);
		int i = 0;
		writer.write("name\tPearson\tinnerP\tdialen\tmaxMed\n");
//		writer.write(",Euc,Cos,Pearson,innerP\n");
		int index = 0;
		while (i < distStrings.size()) {
			String name = distStrings.get(i);
//			rewriter.write(name + "\n");
			double[] distDia = new double[t];
			double[] distMed = new double[t];
			for (int j = 0; j < t; j++) {
				String[] diString = distStrings.get(++i).split("\t");
//				rewriter.write(j + "\t" + distStrings.get(i) + "\n");
				distDia[j] = Double.parseDouble(diString[1]);
				distMed[j] = Double.parseDouble(diString[2]);
			}
			double ps = new PearsonsCorrelation().correlation(distDia, distMed);
			double innerp = Utils.innerProduct(distDia, distMed);
//			double euc = new EuclideanDistance().compute(distDia, distMed);
//			double cos = Utils.cosineSimilarity(distDia, distMed);
//			int isAno = name.startsWith("a")?1:0;
			int diaLan = ((FeatureSequence) pltm.getData().get(index).instances[DIA].getData()).size();
			writer.write(name + "\t" + ps + "\t" + innerp + "\t" + diaLan + "\n");
//			if(i%20 == 0){
//				writer.write("\"" + name + "\"," + euc + "," + cos + "," +  ps + "," + innerp + "\n");
//			}
			i++;
			index++;
		}
//		rewriter.close();
		writer.close();
	}

	private static Map<String, Double> readMedNormFactor(String file) throws IOException {
		Map<String, Double> r = new HashMap<>();
		Files.readAllLines(Paths.get(DIR + file)).stream().forEach(line -> {
			String[] els = line.split("\t");
			r.put(els[0].toLowerCase(), Double.parseDouble(els[1]));
		});
		return r;
	}

	private static void norm(String oldFile, String stdFile, String newFile) throws IOException {
		List<String> oldLines = Files.readAllLines(Paths.get(DIR + oldFile));
		List<String> stdLines = Files.readAllLines(Paths.get(DIR + stdFile));
		BufferedWriter writer = Files.newBufferedWriter(Paths.get(DIR + newFile));
		for (int i = 0; i < oldLines.size(); i++) {
			writer.write(oldLines.get(i));
			String[] els = oldLines.get(i).split("\t");
			if (els.length < 3) {
				writer.newLine();
				continue;
			}
			int elemsCount = els[2].split(" ").length;
			int stdLenght = stdLines.get(i).split("\t")[2].split(" ").length;
			for (int j = 0; j < stdLenght/elemsCount-1; j++) {
				writer.write(" " + els[2]);				
			}
			writer.newLine();
		}
		writer.close();
	}

	private static void InferDoc(String[] files, PolylingualTopicModelMultiReadouts pltm) 
			throws FileNotFoundException {
		InstanceList[] insesList = new InstanceList[files.length]; 
		TopicInferencer[] infs = new TopicInferencer[files.length];
		int docLen = 0;
		for (int i = 0; i < files.length; i++) {
			String fname = files[i];
			insesList[i] = Utils.readInstances(Experiments.DIR + fname);
			infs[i] = pltm.getInferencer(i);
			docLen = insesList[i].size();
		}
		for (int i = 0; i < docLen; i++) {
			System.out.println("Ins " + i);
			for (int j = 0; j < files.length; j++) {
				System.out.print("\tlan:" + j + " ");
				double[] dist = infs[j].getSampledDistribution(insesList[j].get(i), 1000, 10, 200);
				for (double d : dist) {
					System.out.print(d + " ");
				}
				System.out.println();
			}
		}
	}

	public static void EstimateNewModel(String[] inFiles, String filename, int topicNum) throws FileNotFoundException, IOException {
		
		InstanceList[] lans = new InstanceList[inFiles.length];
		
		for (int i = 0; i < lans.length; i++) {
			lans[i] = Utils.readInstances(Experiments.DIR + inFiles[i]);			
		}
		
		PolylingualTopicModelMultiReadouts pltm = new PolylingualTopicModelMultiReadouts(topicNum, 50);

		pltm.addInstances(lans);
		pltm.setNumIterations(1000);
		pltm.showTopicsInterval = 1000;
		pltm.setOptimizeInterval(0);
		

//		topicModel.setTopicDisplay(showTopicsIntervalOption.value, topWordsOption.value);

//		topicModel.setOptimizeInterval(optimizeIntervalOption.value);
		pltm.setBurninPeriod(200);
//		System.out.println(pltm.burninPeriod);
		pltm.estimate(); 
		pltm.write(new File(DIR + filename));
		pltm.printTopWords(new File(DIR + filename + ".tword"), 10, false);
		PrintWriter out = new PrintWriter (new FileWriter ((new File(DIR + filename + ".doctop"))));
		pltm.printDocumentTopics(out, .0, -1);
		out.close();
	}
}