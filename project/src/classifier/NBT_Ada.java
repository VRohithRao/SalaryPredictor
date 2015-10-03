package classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.NBTree;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
/*
 * numIterations -- The number of iterations to be performed.
(see basics of boosting from 
http://en.wikipedia.org/wiki/Ensemble_learning#Bayesian_model_averaging: 
"Boosting involves incrementally building an ensemble by training 
each new model instance to emphasize the training instances that 
previous models misclassified" 
so each base learner is trained on a different and ever-increasingly 
hard-to-classify subset of instances and the way I understood final 
step of boosting is that these models then vote on the class decision, 
whether vote is equal or weighted I do not know)

useResampling -- Whether resampling is used instead of reweighting.
weightThreshold -- Weight threshold for weight pruning.
(reading up on the quoted papers would have this info)
 */
public class NBT_Ada {
	/**
	 * @author yaxin
	 * @param args[0] train.arff path
	 * @param args[1] test.arff path
	 * @param args[2] 1: Use Adaboost; 0: Don't use Adaboost;
	 * @throws Exception 
	 */
	public static void run(String[] args) throws Exception {
		DataSource source = new DataSource(args[0]);
		Instances data = source.getDataSet();
		data.setClassIndex(data.numAttributes() - 1);
		if (args[2].equals("1")) {
			// Use Adaboost	and the following args are options	
			AdaBoostM1 adaboost = new AdaBoostM1();
			NBTree nbtModel = new NBTree();
			adaboost.setClassifier(nbtModel);
			adaboost.buildClassifier(data);
			
			// Evaluation:
			Evaluation eval = new Evaluation(data);
			Instances testData = new DataSource(args[1]).getDataSet();
			testData.setClassIndex(testData.numAttributes() - 1);
			eval.evaluateModel(adaboost, testData);
			//System.out.println(adaboost.toString());
			System.out.println(eval.toSummaryString("\nNBTree Results with "
					+ "AdaBoost\n==============\n", false));
			System.out.println("======\nConfusion Matrix:");
			double[][] confusionM = eval.confusionMatrix();
			for (int i = 0; i < confusionM.length; ++i) {
				for (int j = 0; j < confusionM[i].length; ++j) {
					System.out.format("%10s ", confusionM[i][j]);
				}
				System.out.print("\n");
			}
		}
		else {
			// Don't use Adaboost
			NBTree model = new NBTree();
			model.buildClassifier(data);
			
			// Evaluation:
			Evaluation eval = new Evaluation(data);
			Instances testData = new DataSource(args[1]).getDataSet();
			testData.setClassIndex(testData.numAttributes() - 1);
			eval.evaluateModel(model, testData);
			//System.out.println(model.toString());
			System.out.println(eval.toSummaryString("\nNBTree Results "
					+ "without boosting\n======\n", false));
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
}