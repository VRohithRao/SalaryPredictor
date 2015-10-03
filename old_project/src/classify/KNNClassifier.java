package classify;
/*****************************************************************
 * Valid options are:

 -I
  Weight neighbours by the inverse of their distance
  (use when k > 1)
 -F
  Weight neighbours by 1 - their distance
  (use when k > 1)
 -K <number of neighbors>
  Number of nearest neighbours (k) used in classification.
  (Default = 1)
 -E
  Minimise mean squared error rather than mean absolute
  error when using -X option with numeric prediction.
 -W <window size>
  Maximum number of training instances maintained.
  Training instances are dropped FIFO. (Default = no window)
 -X
  Select the number of nearest neighbours between 1
  and the k value specified using hold-one-out evaluation
  on the training data (use when k > 1)
 -A
  The nearest neighbour search algorithm to use 
  (default: weka.core.neighboursearch.LinearNNSearch).
 */

import java.util.Arrays;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class KNNClassifier {
	public static void run(String[] args) throws Exception {
		/***************************************************
		 * @param args[0...length-3] options
		 * @param args[length-2] train.arff
		 * @param args[length-1] test.arff
		 */
		String[] options = Arrays.copyOfRange(args, 0, args.length-2);
		DataSource source = new DataSource(args[args.length-2]);
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		IBk model = new IBk();
		model.setOptions(options);
		model.buildClassifier(data);
		
		// Evaluation:
		Evaluation eval = new Evaluation(data);
		Instances testData = new DataSource(args[args.length-1]).getDataSet();
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