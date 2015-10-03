package classify;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class SMOClassifier {
	public static void run(String[] args) throws Exception {
		/***************************************************
		 * @param args[0]: train arff path
		 * @param args[1]: test arff path
		 */
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		SMO model = new SMO();
		model.buildClassifier(data);
		
		// Evaluation:
		Evaluation eval = new Evaluation(data);
		Instances testData = new DataSource(args[1]).getDataSet();
		testData.setClassIndex(testData.numAttributes() - 1);
		eval.evaluateModel(model, testData);
		System.out.println(model.toString());
		System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		System.out.println("======\nConfusion Matrix:");
		double[][] confusionM = eval.confusionMatrix();
		for (int i = 0; i < confusionM.length; ++i) {
			for (int j = 0; j < confusionM[i].length; ++j) {
				System.out.format("%10s ", confusionM[i][j]);
			}
			System.out.print("\n");
		}
	}
}