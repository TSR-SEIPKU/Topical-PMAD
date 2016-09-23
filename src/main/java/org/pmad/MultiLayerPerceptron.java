package org.pmad;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class MultiLayerPerceptron {
	
	BasicNetwork network = null;
	
	public MultiLayerPerceptron(double[][] testMatrixA, double[][] testMatrixB) {
		System.out.println("trainning nn...");
		int n = testMatrixA[0].length;
		int m = testMatrixB[0].length;
		network = new BasicNetwork();
		network.addLayer(new BasicLayer(new ActivationLinear(), true, n));
//		network.addLayer(new BasicLayer(new ActivationTANH(), true, n));
		network.addLayer(new BasicLayer(new ActivationLinear(), true, m));
		network.getStructure().finalizeStructure();
		network.reset();
		
		MLDataSet train = new BasicMLDataSet(testMatrixA, testMatrixB);
		
		
		final ResilientPropagation rsp = new ResilientPropagation(network, train);
		int i = 0;
		double laste = Double.MAX_VALUE;
		double err = Double.MAX_VALUE;
		do {
			laste = err;
			rsp.iteration();
			err = rsp.getError();
			i++;
			if (i%100 == 0) {
				System.out.println(i + "\t" + err + "\t" + laste);
			}
		} while(err < laste || i < 1000);
		rsp.finishTraining();
		System.out.println(i + "\t" + err + "\t" + laste) ;
		double[] output = new double[m];
		network.compute(testMatrixA[0], output);
//		for (int i = 0; i < output.length; i++) {
//			System.out.println(output[i] + "\t" + testMatrixB[0][i]);
//		}
		System.out.println("Done!");
		Encog.getInstance().shutdown();
	}
	
	public MultiLayerPerceptron(double[][] testMatrixA, double[][] testMatrixB, boolean forBaseline) {
		System.out.println("trainning nn...");
		int n = testMatrixA[0].length;
		int m = testMatrixB[0].length;
		network = new BasicNetwork();
		network.addLayer(new BasicLayer(new ActivationLinear(), true, n));
		network.addLayer(new BasicLayer(new ActivationTANH(), true, n));
		network.addLayer(new BasicLayer(new ActivationLinear(), true, m));
		network.getStructure().finalizeStructure();
		network.reset();
		
		MLDataSet train = new BasicMLDataSet(testMatrixA, testMatrixB);
		
		
		final ResilientPropagation rsp = new ResilientPropagation(network, train);
		int i = 0;
		double laste = Double.MAX_VALUE;
		double err = Double.MAX_VALUE;
		do {
			laste = err;
			rsp.iteration();
			err = rsp.getError();
			i++;
			if (i%100 == 0) {
				System.out.println(i + "\t" + err + "\t" + laste);
			}
		} while(err < laste || i < 1000);
		rsp.finishTraining();
		System.out.println(i + "\t" + err + "\t" + laste) ;
		double[] output = new double[m];
		network.compute(testMatrixA[0], output);
//		for (int i = 0; i < output.length; i++) {
//			System.out.println(output[i] + "\t" + testMatrixB[0][i]);
//		}
		System.out.println("Done!");
		Encog.getInstance().shutdown();
	}

	public double[] fit(double[] distA) {
		double[] res = new double[distA.length];
		network.compute(distA, res);
		return res;
	}
	
	

}
