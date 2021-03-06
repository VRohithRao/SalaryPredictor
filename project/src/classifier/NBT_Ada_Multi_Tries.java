package classifier;
import java.util.Arrays;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.NBTree;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
public class NBT_Ada_Multi_Tries {
	private static final int ROUNDS_NUM = 50;
	private static final int ITER_NUM = 20;
	/**
	 * @author yaxin
	 * @param args[0] train.arff path
	 * @param args[1] test.arff path
	 * @throws Exception 
	 */
	public static void run(String[] args) throws Exception {
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		Random random = new Random(1111);

		double sum_of_err = 0;
		// I found that there is no difference in error rate for each round.
		// Even changing the random seed.
		for (int i = 0; i < ROUNDS_NUM; ++i) {
			AdaBoostM1 adaboost = new AdaBoostM1();
			NBTree nbtModel = new NBTree();
			adaboost.setClassifier(nbtModel);
			adaboost.setOptions(new String[] {"-S", 
					new Integer(random.nextInt()).toString(),
					"-I", new Integer(ITER_NUM).toString()});

			adaboost.buildClassifier(data);
			System.out.println("Current options of the classifier: " + 
					Arrays.toString(adaboost.getOptions()));

			// Evaluation:
			Evaluation eval = new Evaluation(data);
			Instances testData = new DataSource(args[1]).getDataSet();
			testData.setClassIndex(testData.numAttributes() - 1);
			eval.evaluateModel(adaboost, testData);
			System.out.println("Error rate of round " + i + " : " + eval.errorRate());
			sum_of_err += eval.errorRate();
		}
		System.out.println("AVERAGE ERROR RATE: " + sum_of_err/ROUNDS_NUM);
	}
}