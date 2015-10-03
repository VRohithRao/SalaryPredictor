import java.util.Arrays;

import classify.C45DTClassifier;
import classify.ID3Classifier;
import classify.KNNClassifier;
import classify.LogisticRegClassifier;
import classify.NaiveBayesClassifier;
import classify.SMOClassifier;
import trans_to_arff.TransTestArff;
import trans_to_arff.TransTrainArff;

public class MainDriver {
	public static void main (String[] args) throws Exception {
		/**************************************************
		 * Now the KNN implementation is the only with options supported.
		 * @Usage:
		 * 1. Translate the training input to arff
		 * java -jar <jar-name> -transtrain <input-file-path> <output-path>
		 * 
		 * 2. Translate the testing input to arff
		 * java -jar <jar-name> -transtest <input-file-path> <outupt-path>
		 * 
		 * 3. Train with Naive Bayes with train and test data prepared:
		 * java -jar <jar-name> -nb <train-arff> <test-arff>
		 * 
		 * 4. Train with KNN:
		 * java -jar <jar-name> -knn [OPTIONS] <train-arff> <test-arff>
		 * 
		 * 5. Train with ID3 decision tree:
		 * java -jar <jar-name> -id3 <train-arff> <test-arff>
		 * 
		 * 6. Train with C4.5 decision tree:
		 * java -jar <jar-name> -c45 <train-arff> <test-arff>
		 * 
		 * 7. Train with Logistic Regression Classifier:
		 * java -jar <jar-name> -logreg <train-arff> <test-arff>
		 * 
		 * 8. Train with SMO:
		 * java -jar <jar-name> -smo <train-arff> <test-arff>
		 */
		if (args[0].equals("-transtrain")) {
			if (args.length != 3) {
				//TODO: print usage error
				System.exit(1);
			}
			TransTrainArff.main(Arrays.copyOfRange(args, 1, 3));
		}
		else if (args[0].equals("-transtest")) {
			if (args.length != 3) {
				//TODO: print usage error
				System.exit(1);
			}
			TransTestArff.main(Arrays.copyOfRange(args, 1, 3));
		}
		else if (args[0].equals("-nb")) {
			if (args.length != 3) {
				//TODO: print usage error
				System.exit(1);
			}
			NaiveBayesClassifier.run(Arrays.copyOfRange(args, 1, 3));
		}
		else if (args[0].equals("-knn")) {
			KNNClassifier.run(Arrays.copyOfRange(args, 1, args.length));
		}
		else if (args[0].equals("-id3")) {
			ID3Classifier.run(Arrays.copyOfRange(args, 1, args.length));
		}
		else if (args[0].equals("-c45")) {
			C45DTClassifier.run(Arrays.copyOfRange(args, 1, args.length));
		}
		else if (args[0].equals("-logreg")) {
			LogisticRegClassifier.run(Arrays.copyOfRange(args, 1, args.length));
		}
		else if (args[0].equals("-smo")) {
			SMOClassifier.run(Arrays.copyOfRange(args, 1, args.length));
		}
		else {
			// TODO: print usage error
			System.exit(1);
		}
	}
}