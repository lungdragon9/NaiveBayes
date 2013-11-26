package Project.NaiveBayes;

import weka.classifiers.bayes.NaiveBayes;

import java.util.Arrays;

import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.GainRatioSplitCrit;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.PathInfo;
import weka.core.converters.ConverterUtils.DataSource;

@SuppressWarnings("unused")
public class NaiveBayesProject {

    public static void main(String argsp[])
    {
    	//Catches all exceptions
        try {
            Instances tmpInstances = null;

            //Finds data
            NaiveBayes model = new NaiveBayes();
            tmpInstances = (new DataSource("https://dl.dropboxusercontent.com/u/88175668/binary/colic/colic.arff")).getDataSet();
		
            double[] weights = new double[tmpInstances.numAttributes()];
            
            
            Arrays.fill(weights, 1);
            //For testing only
            //model.setWeight(weights);
            
            //Sets the cutoff point in the data
            int cutoff = (int)(tmpInstances.numInstances() * .8);
            
            //Creates both the train and test data
            Instances train = new Instances(tmpInstances,0,cutoff);
            Instances test = new Instances(tmpInstances,cutoff,tmpInstances.numInstances() -cutoff);

            
            //Sets the class
            train.setClass(train.attribute(train.numAttributes()-1));
            test.setClass(test.attribute(train.numAttributes()-1));

          //Builds the Naive Bayes Model
            model.setWeight(weights);
            model.buildClassifier(train);
            
            
            //Tests the model
            //Evaluation eval = new Evaluation(test);
           // eval.evaluateModel(model,test);

            //Output
            //System.out.println("Num Testing Instances " + test.numInstances());
            //System.out.println("Correct: " + eval.correct());
            //System.out.println("Incorrect " + eval.incorrect());
            //System.out.println("Error rate " + eval.errorRate());
            //System.out.println("Pct Correct " +eval.pctCorrect());
            //System.out.println("done");
            
            
            //Hill Climbing
            weights = GainRatio(train);
            //weights = MarkovChain(train);
            weightDisplay(weights);
            //Arrays.fill(weights, 1);
            weights = HillClimbing(train,weights,10,.000001);
            weightDisplay(weights);
            
            //Builds the Naive Bayes Model
            model.setWeight(weights);
            
            //Tests the model
            Evaluation eval = new Evaluation(test);
            eval.evaluateModel(model,test);

            //Output
            System.out.println("Num Testing Instances " + test.numInstances());
            System.out.println("Correct: " + eval.correct());
            System.out.println("Incorrect " + eval.incorrect());
            System.out.println("Error rate " + eval.errorRate());
            System.out.println("Pct Correct " +eval.pctCorrect());
            System.out.println("done");

        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
    
    /**
     * Finds the weights using Gain Ratio
     * @param train : training set
     * @return : returns the weights
     */
    public static double[] GainRatio(Instances train)
    {
    	double[] weight = new double[train.numAttributes()];
    	double sum=0;
    	try {
    		InfoGainAttributeEval Gain = new InfoGainAttributeEval();
	    	
	    	//Trains the GainRatio
			Gain.buildEvaluator(train);
	    	
			//Finds the weights using the Gain Ratio
	    	for(int i =0; i < train.numAttributes(); i ++)
	    	{
	    		weight[i] = Gain.evaluateAttribute(i);
	    	}
    	
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	
		return weight_Distributions(weight);
    }
	
    /**
     * Finds the weights using Hill Climbing
     * @param train : training set
     * @param weights[] : Used for combines methods
     * @return : returns the weights
     * @throws Exception 
     */
    public static double[] HillClimbing(Instances train,double weights[], double learningRate, double AOCCheck) throws Exception
    {

    	Evaluation eval;
    	ThresholdCurve curvefinder = new ThresholdCurve();
    	double oAOC =0;
    	//Can be switched to just an double without the array
    	double weigh_change[] = new double[train.numAttributes()];
    	double AUCPost = 0;
    	double AUC=0;
    	boolean firstCheck = true;
    	NaiveBayes model = new NaiveBayes();
    	
    	
    	
    	//Splits the training set into two smaller sets
    	int cutoff = (int)(train.numInstances() * .2);
        
        //Creates both the train and test data
        Instances train_HC = new Instances(train,0,cutoff);
        //Helps to validate the changes to the weights
        Instances validate_HC = new Instances(train,cutoff,train.numInstances() -cutoff);
    	
		model.buildClassifier(train_HC);        
		
		eval = new Evaluation(train_HC);
		model.setWeight(weights);
		eval.evaluateModel(model, validate_HC);
		
		//System.out.println(eval.pctCorrect());
		
		for(int i =0; i < train_HC.numAttributes(); i++)
		{
			
    		while(true)
    		{
    			//Sets weights builds model
    			eval = new Evaluation(train_HC);
    			model.setWeight(weights);
    			model.buildClassifier(train_HC); 
        		eval.evaluateModel(model, validate_HC);
        		
        		AUC = ThresholdCurve.getROCArea(curvefinder.getCurve(eval.predictions()));
        		System.out.println(AUC);
        		
        		//Finds o(AUC)
    			oAOC = 1/(1+Math.pow(Math.E,(-1*AUC)));
    			weigh_change[i] = learningRate*oAOC*Math.pow((1-oAOC),2);
    			
				weights[i] -= weigh_change[i];
    			
				eval = new Evaluation(train_HC);
    			model.setWeight(weights);
    			model.buildClassifier(train_HC); 
        		eval.evaluateModel(model, validate_HC);
        		
        		AUCPost = ThresholdCurve.getROCArea(curvefinder.getCurve(eval.predictions()));
    			//System.out.println(AUCPost);
    			//Should check to see if the change in AUC is enough to keep going
    			//Make this a different method and break if no AUC gain
        		//System.out.println((AUC - AUCPost) > AOCCheck);
        		
	    			if((AUCPost - AUC) < AOCCheck)
	    			{
	    				weights[i] += weigh_change[i];
	
	            		break;
	        			
	    			}
	    			if(weights[i] <=0)
	    			{
	    				weights[i] =0;
	    				break;
	    			}

    		}
		}
    	
		return weight_Distributions(weights);
    	
    }
    
    /**
     * Finds the weights using Markov Chain
     * @param train : training set
     * @return : returns the weights
     */
    public static double[] MarkovChain(Instances train)
    {
    	double[] weight = new double[train.numAttributes()];
    	double sum=0;
    	try {
	
	    	//Trains the MarkovChain
			//Gain.buildEvaluator(train);
	    	
			//Finds the weights using the Gain Ratio
	    	for(int i =0; i < train.numAttributes(); i ++)
	    	{
	    		//weight[i] = Gain.evaluateAttribute(i);
	    		sum += weight[i];
	    	}
	    	//Helps the weights
	    	for(int i =0; i < train.numAttributes(); i ++)
	    	{
	    		weight[i] = (weight[i] * train.numAttributes())/sum;
	    	}
    	
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return weight;
    }
    
    /**
     * A method mostly used to testing
     * @param weights
     * @return
     */
    public static double[] weight_Distributions(double[] weights)
    {
    	double sum =0;
    	
    	for(int i =0; i < weights.length; i ++)
    	{	
    		sum += weights[i];    		
    	}
    	for(int i =0; i < weights.length; i ++)
    	{	
    		weights[i] = (weights[i] * weights.length) / sum;    		
    	}
    	return weights;
    }
    
    
    //Please use this MCMC Eli
    //Most of the code should come out of Hill Climbing
    // This to find the AUC
    ///AUC = ThresholdCurve.getROCArea(curvefinder.getCurve(eval.predictions()));
    /**
     * Finds the weights using Markov Chain Monte Carlo
     * @param train : training set
     * @param weights[] : Used for combines methods
     * @return : returns the weights
     */
    public double[] MCMC(Instances train,double weights[])
    {
    	
		return weights;
    	
    }
    
    /**
     * Diplays the weights
     * @param weights
     */
    public static void weightDisplay(double[] weights)
    {
    	int i =0;
    	for(i =0; i < weights.length; i ++)
    	{
    		System.out.print(weights[i] + " ");
    	}
    	System.out.println();
    }
}
