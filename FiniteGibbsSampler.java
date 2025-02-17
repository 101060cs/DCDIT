package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;  
import java.util.LinkedList;  
import java.util.List;  
import java.util.Collections;
import java.util.HashMap;
import java.util.Random;
import java.math.*;
import java.text.DecimalFormat;

import org.rosuda.JRI.REXP;
import org.rosuda.JRI.RList;
import org.rosuda.JRI.RVector;
import org.rosuda.JRI.RMainLoopCallbacks;
import org.rosuda.JRI.Rengine;
//import java.de.uni_leipzig.informatik.asv.utils.*;

import myutils.InputOutputUtils;
import myutils.IntegerPair;
import myutils.specialFunctions;
import myutils.LinkedTreeNode;
import training.SingleRengine;
import datastructures.DataPaths;
import datastructures.Document;
import datastructures.Link;
import datastructures.UserProfile;

public class FiniteGibbsSampler {
  private static final double Infinity = 0;
/**
   * Hyper
   */

	// user-community 
  	public static double rho = 0.01;
  	// CR 
	public static double lambda = 0.01;
	//community-topic-level
	public static double alpha = 0.01;
	//observed-topic
	public static double epsilon= 0.1;
	// topic-word
	public static double beta = 0.1;
	// comm-topic-user
	public static double delta = 0.01;
	// Sums
	//public static double lambdaSum;
	public static double alphaSum;
	public static double epsilonSum;
	public static double betaSum;
	public static double rhoSum;
	public static double deltaSum;

  /**
   * Parameters
   */
  public int K; //var  
  public int V;
  public int U;
  public int C;
  public int L;
  public int linkNum;
  public int t; 

  /**
   * Data 
   */
  public HashMap<Integer, UserProfile> userProfile;
    
  /**
   * Counts
   */
  // user comm level
  public int[][] numberOfCommByUser; // n_{u}^{c} = [u][c]
  public int[] TotNumberOfCommByUser; // n_{u}^{.} = [u]  
  
  // comm comm
  public int[][] numberOfCommByComm; // n_{c}^{c} = [c][c]
  public int[] TotNumberOfCommByComm; // n_{c}^{.} = [c] 
  
  //comm - treeLevel 
  public int[][] numberOfTreeLevelByComm;// n_{c}^{l} = [c][l]
  public int[] TotNumberOfTreeLevelByComm; // n_{c}^{.} = [c]

  // topic level
  public int[][] numberOfWordByTopic; // n_{k}^{v} = [k][v]
  public int[] TotNumberOfWordByTopic; // n_{k}^{.} = [k]
  
  // comm labeled topic level 
  public int[][] numberOfLabeledTopicByComm; // n_{c}^{0,1,2} = [c][0,1,2]
  public int[] TotNumberOfLabeledTopicByComm; // n_{c} = [c] 
  
  // comm unobserved topic level 
  LinkedList<LinkedTreeNode> topicTree;
 
  //user doc topic path
  //public int [][][] userTopicPath;// [u][d][l]
  /**
   * Probabilities
   */
  //[u][c]: user - comm
  double[][] userCommDistribution;	
  //[c][c]: comm - comm
  public double[][] comCommDistribution;
  //[c][l]: comm - Treelevel
  public double[][] commTreeLevelDistribution;
  // [k][v]: topic - word
  public double[][] topicWordDistribution;
  // [c][0,1]: comm -labeled topic
  double[][] commLabeledTopicDistribution;


  		
  //[u][c]
  public double[][] userCommSum;
  //[c][c]
  public double[][]  comCommSum;	
  //[c][l]
  public double[][]  comTreeLevelSum;
  // [k][v]: topic - word
  public double[][] topicWordSum;
//[c][0,1]
  public double[][] commLabeledTopicSum;

  public int sumNum = 0;
  
  /**
   * Gibbs sampler configuration
   */
  public Random random = new Random();
  public int ITERATIONS;
  public int SAMPLE_LAG;
  public int SAMPLE_PER_LAG;
  public int OUTPUT_LAG;
  public int LIKELIHOOD_LAG;
  public int BURN_IN;
  public String outputPath; 
  public int numOfMarkov;
  public int numOfExp;
  
  /**
   * Temp 
   */
  double[] z_prior;
  double[] za_prior_fixedA;
  double[] za_prior_fixedK;
  double[] z_post_w;
  BigDecimal[]  z_post_w_bd;
  BigDecimal[]  z_post_w_bd_fixedk;
  double[] z_post_t;
  double z_post_w_min;
  double z_post_t_min;
/*  double[] z_post_t;
  double z_post_w_min;
  double z_post_t_min;*/
  int[] z_post_w_carrier;
  double threshold = 1E-160; 
  double multiplier = 1E160; 


  /**
   * For computing likelihood
   */
  HashMap<Integer, Double> likelihood_final 
    = new HashMap<Integer, Double>();
  HashMap<Integer, Integer> likelihood_final_carrier 
    = new HashMap<Integer, Integer>();
  HashMap<Integer, Double> likelihood_text 
    = new HashMap<Integer, Double>();
  HashMap<Integer, Integer> likelihood_text_carrier 
    = new HashMap<Integer, Integer>();
  HashMap<Integer, Double> likelihood_link 
    = new HashMap<Integer, Double>();
  HashMap<Integer, Integer> likelihood_link_carrier 
    = new HashMap<Integer, Integer>();
  HashMap<Integer, Double> likelihood_commTreeLevel 
  = new HashMap<Integer, Double>();
HashMap<Integer, Integer> likelihood_commTreeLevel_carrier 
  = new HashMap<Integer, Integer>();
  ArrayList<Integer> likelihood_iters = new ArrayList<Integer>();  
  public double[][] logTopicWordDistribution;

  /**
   * 
   * @param userProfile[]: dataset
   */
  public void addInstances(HashMap<Integer, UserProfile> userProfile, int C, int t,int V, int U, int iters, int sample_lag, int sample_per_lag,
      int output_lag, int likelihood_lag, int burn_in, String outputPath,int _numOfMarkov, int _numOfExp,
      double _inlambda, double _inalpha,  double _inbeta,double _inEpsilon, double _inRho,double _inDelta,int _inLinkNum, int L) {
    this.userProfile = userProfile;
    //this.K = K;
    this.V = V;
    this.U = U;    
    this.C = C;
    this.L=L;
    this.t=t;
    this.lambda=_inlambda;
    this.alpha=_inalpha;
    this.beta=_inbeta;
    this.epsilon=_inEpsilon;
    this.rho=_inRho;
    this.delta=_inDelta;
    this.linkNum=_inLinkNum;
    //this.gamma1=gamma1;
    //this.gamma2=gamma2;
  //read gamma
    //readGamma(t);
    ITERATIONS = iters;
    SAMPLE_LAG = sample_lag;
    SAMPLE_PER_LAG=sample_per_lag;
    OUTPUT_LAG = output_lag;
    LIKELIHOOD_LAG = likelihood_lag;
    BURN_IN = burn_in;
    numOfExp=_numOfExp;
    numOfMarkov=_numOfMarkov;
    this.outputPath = outputPath;
    
    		

	
    /**
     * Counts
     */
	
	//user comm
    numberOfCommByUser= new int [U][C]; 
    TotNumberOfCommByUser=new int[U];  

    // comm comm
    numberOfCommByComm= new int [C][C];  // n_{c}^{u} = [c][u]
    TotNumberOfCommByComm= new int [C]; // n_{c}^{.} = [c] 
    
    //comm - treeLevel 
    numberOfTreeLevelByComm= new int [C][L];// n_{c}^{l} = [c][l]
    TotNumberOfTreeLevelByComm= new int [C]; // n_{c}^{.} = [c]

   
    
    // comm labeled topic level 
    numberOfLabeledTopicByComm= new int [C][3]; // n_{c}^{0,1,2} = [c][0,1,2]
    TotNumberOfLabeledTopicByComm= new int [C]; // n_{c}^{0,1}^{.} = [c] 
    
    topicTree= new LinkedList<LinkedTreeNode>();
  //user topic path
   //userTopicPath= new int [U][L];// [c][l]
    //[u][c]
    userCommSum = new double[U][C];
    //[c][c]
    comCommSum = new double[C][C];	
    //[c][l]
    comTreeLevelSum = new double[C][L];
   
  //[c][0,1]
    commLabeledTopicSum = new double[C][3];			
   
    
    /**
     * Probabilities
     */
    // [u][c]: user - comm
 	userCommDistribution = new double[U][C];
 	
 	  //[c][c]: comm - comm
 	comCommDistribution = new double[C][C];
 	  //[c][l]: comm - Treelevel
 	 commTreeLevelDistribution = new double[C][L];
 	  
 	  // [c][0,1]: comm -labeled topic
 	 commLabeledTopicDistribution = new double[C][3];
 			
 			//lambdaSum = lambda * L;// each user is different
 			alphaSum = alpha*L;
 			// topic-word
 			betaSum = beta * V;
 			epsilonSum =epsilon*2; 			
 			rhoSum=rho*U;
 			deltaSum=delta*C;
   
  }
  
  public int calNumberOfLinks(){
    int numberOfLinks = 0;
    for(int i=0; i<U; i++){
      numberOfLinks += userProfile.get(i).getLinkSize();
    }
    return numberOfLinks;
  }
     
  /**
   * Initialize
   */
  public void initialState(){
	    UserProfile curUser;
	    //topic tree
	    //root
	    //LinkedTreeNode root = new LinkedTreeNode("root",null,C,0);
	    topicTree.add(new LinkedTreeNode("root",null,C,0,0));
	    topicTree.add(new LinkedTreeNode("SARS",topicTree.get(0),C,1,1));
	    topicTree.add(new LinkedTreeNode("COVID",topicTree.get(0),C,2,1));
	    topicTree.add(new LinkedTreeNode("Other",topicTree.get(0),C,3,1));
	    int topicNodeID=4;
	    for(int i=0; i<U; i++){
	      curUser = userProfile.get(i);	  
	      
	      for(int link_idx=0; link_idx < curUser.links.size(); link_idx++){	    	 
	        int l1_ran_k,ran_l,ran_c,ran_tc,ran_sc;	
	        int tempTargetid=curUser.getLinkTargetId(link_idx);
	        ran_c= random.nextInt(C);	
	        curUser.setLinkSourceC(link_idx, ran_c);
	        ran_tc= random.nextInt(C);	
	        curUser.setLinkTargetC(link_idx, ran_tc);
	        numberOfCommByUser[i][ran_c]++;
	        TotNumberOfCommByUser[i]++;	   
	        numberOfCommByUser[tempTargetid][ran_tc]++;
	        TotNumberOfCommByUser[tempTargetid]++;	   
	        numberOfCommByComm[ran_c][ran_tc]++;
	        TotNumberOfCommByComm[ran_c]++;
	  
	        //Topic
	        curUser.setLinkTopicLevelK(link_idx, 0, topicTree.get(0));	        
	        l1_ran_k= random.nextInt(3);  
	        curUser.setLinkTopicLevelK(link_idx, 1, topicTree.get(l1_ran_k+1));
	        numberOfLabeledTopicByComm[ran_c][l1_ran_k]++;
	        TotNumberOfLabeledTopicByComm[ran_c]++;
	        
	        for(int l=2;l<L;l++)
	        {
	        	LinkedTreeNode parentNode= curUser.getLinkTopicLevel(link_idx,l-1);
	        	
	        	if(parentNode.getChilden().isEmpty())
	        		{
	        			LinkedTreeNode newTopicNode = new LinkedTreeNode("level:"+l,parentNode,C,topicNodeID,l);
	        			newTopicNode.numCommNodes[ran_c]++;
	        			newTopicNode.TotNumCommNodes++;
	        			curUser.getLinkTopicLevel(link_idx,l-1).addChild(newTopicNode);
	        			topicTree.add(newTopicNode);
	        			curUser.setLinkTopicLevelK(link_idx, l,newTopicNode);
	        			topicNodeID++;
	        		}
	        	else 
	        	{
	        		List<LinkedTreeNode> curNodeList= parentNode.getChilden();
	        		double p[] = new double[curNodeList.size()+1]; 
	        		double sum=0;
	        		for(int no=0;no<curNodeList.size();no++)
	        		{
	        			sum+=(double)(curNodeList.get(no).numCommNodes[ran_c]*10+curNodeList.get(no).TotNumCommNodes);
	        		}
	        		for(int no=0;no<curNodeList.size();no++)
	        		{
	        			p[no]=(curNodeList.get(no).numCommNodes[ran_c]*10+curNodeList.get(no).TotNumCommNodes)/(sum+lambda);	        			
	        		}
	        		p[curNodeList.size()]=lambda/(sum+lambda);
	        		
	        		for(int pp=1; pp<curNodeList.size()+1; pp++)
	        	        p[pp] += p[pp-1];
	        	    double ran = random.nextDouble() * p[curNodeList.size()];
	        	    int k_new;
	        	    for(k_new=0; k_new<curNodeList.size()+1; k_new++){
	        	      if(ran < p[k_new])
	        	        break;
	        	    }
	        	    if(k_new == curNodeList.size())//new topic node
	        	    {
	        	    	LinkedTreeNode newTopicNode = new LinkedTreeNode("level:"+l,parentNode,C,topicNodeID,l);
	        			newTopicNode.numCommNodes[ran_c]++;
	        			newTopicNode.TotNumCommNodes++;
	        			parentNode.addChild(newTopicNode);
	        			topicTree.add(newTopicNode);
	        			curUser.setLinkTopicLevelK(link_idx, l,newTopicNode);
	        			topicNodeID++;
	        	    }
	        	    else //existing topic node
	        	    {
	        	    	parentNode.getChilden().get(k_new).numCommNodes[ran_c]++;
	        	    	parentNode.getChilden().get(k_new).TotNumCommNodes++;
	        	    	curUser.setLinkTopicLevelK(link_idx, l,parentNode.getChilden().get(k_new));
	        	    }
	        		
	        	}
	        }

	        ran_l= random.nextInt(L);	
	        curUser.setLinkL(link_idx, ran_l);
	        numberOfTreeLevelByComm[ran_c][ran_l]++;
	        TotNumberOfTreeLevelByComm[ran_c]++;	   
    
	        }	      	

	    }
	    // topic level
	    K=topicNodeID;
	    numberOfWordByTopic= new int [K][V]; // n_{k}^{v} = [k][v]
	    TotNumberOfWordByTopic= new int [K];  // n_{k}^{.} = [k]
	    // [k][v]: topic - word
	    topicWordSum = new double[K][V];
	 // [k][v]: topic - word
	 	 topicWordDistribution = new double[K][V];
	 	 /**
	      * Temp 
	      */
	     z_prior = new double[K];
	     za_prior_fixedA = new double[K];
	     za_prior_fixedK = new double[3];
	     z_post_w = new double[K];
	     z_post_w_bd = new BigDecimal[K];
	     z_post_w_bd_fixedk= new BigDecimal[3];
	     z_post_t = new double[K];
	     z_post_w_carrier = new int[K];
	     logTopicWordDistribution = new double[K][V];
	     
	    for(int i=0; i<U; i++){
		      curUser = userProfile.get(i);	  		      
		      for(int link_idx=0; link_idx < curUser.links.size(); link_idx++){	    	 
		       int K= curUser.getLinkTopicLevel(link_idx, curUser.getLinkL(link_idx)).getID();
		        Document linkContent= curUser.links.get(link_idx).linkDoc;
		        for(int v : linkContent.word_count.keySet()){//主题对应edge中每个word+其出现次数
		          	try{
		          		numberOfWordByTopic[K][v] +=linkContent.word_count.get(v);
		          		TotNumberOfWordByTopic[K] += linkContent.word_count.get(v);
		          	}
		          	catch(Exception e)
		          	{
		          		 System.out.println("word Error! ");
		          	}
		          }	
		      }
		      }
	  }
	 
  /**
   * Step one step ahead
   */
  public void nextGibbsSweep() {
    UserProfile curUser;
    int y_link_new,s_link_new,c_ii_link_new,z_new;   
    //Sample C 
    for(int i=0; i<U; i++){
      curUser = userProfile.get(i);
      // only sample existing links
      for(int link_idx=0; link_idx < curUser.links.size(); link_idx++){
        //Link cur_link = curUser.links.get(link_idx);
        //int ii = cur_link.ii;
        int c_new = sampleC(i, link_idx);//sampleC(i, ii, link_idx)
        curUser.setLinkSourceC(link_idx, c_new);
      }
    }

    
    // Sample l
    for(int i=0; i<U; i++){
        curUser = userProfile.get(i);
        // only sample existing links
        for(int link_idx=0; link_idx < curUser.links.size(); link_idx++){         
          int l_new = sampleL(i, link_idx);//sampleC(i, ii, link_idx)
          curUser.setLinkL(link_idx, l_new);
        }
      }
    //Sample z
    for(int i=0; i<U; i++){
      curUser = userProfile.get(i);
      // only sample existing links
      for(int link_idx=0; link_idx < curUser.links.size(); link_idx++){
        Link cur_link = curUser.links.get(link_idx);
        Document linkContent= curUser.links.get(link_idx).linkDoc;
        int c = cur_link.c_i;
		/*
		 * Document linkContent=cur_link.linkDoc; if(linkContent.length>0) { y_link_new
		 * = sampleLinkY(i, link_idx); curUser.setLinkContentY(link_idx, y_link_new); }
		 */
        int k_old = curUser.getLinkTopicLevel(link_idx, 1).getID();
        numberOfLabeledTopicByComm[c][k_old-1]--;
	    TotNumberOfLabeledTopicByComm[c]--; 
	    int kk_old = curUser.getLinkTopicLevel(link_idx,curUser.getLinkL(link_idx)).getID();
	        for(int v : linkContent.word_count.keySet()){//主题对应edge中每个word+其出现次数
	          	try{
	          		numberOfWordByTopic[kk_old][v] -=linkContent.word_count.get(v);
	          		TotNumberOfWordByTopic[kk_old] -= linkContent.word_count.get(v);
	          	}
	          	catch(Exception e)
	          	{
	          		 System.out.println("word Error! ");
	          	}
	          }
    	  double p1[] = new double[3]; 
    	  for(int pp1=0;pp1<3;pp1++)
    	  {
    		  p1[pp1]=(numberOfLabeledTopicByComm[c][pp1]+epsilon)/(TotNumberOfLabeledTopicByComm[c]+epsilonSum);    		  
    	  }
    	  for(int pp=1; pp<3; pp++)
  	        p1[pp] += p1[pp-1];
  	    double ran = random.nextDouble() * p1[2];
  	    int k_new;
  	    for(k_new=0; k_new<3; k_new++){
  	      if(ran < p1[k_new])
  	        break;
  	    }
  	    curUser.setLinkTopicLevelK(link_idx, 1, topicTree.get(k_new+1));    
	    numberOfLabeledTopicByComm[c][k_new]++;
	    TotNumberOfLabeledTopicByComm[c]++;
	    
        for(int l=2;l<L;l++)
        {
        	LinkedTreeNode parentNode= curUser.getLinkTopicLevel(link_idx,l-1);
        	
        		List<LinkedTreeNode> curNodeList= parentNode.getChilden();
        		double p[] = new double[curNodeList.size()]; 
        		double sum=0;
        		for(int no=0;no<curNodeList.size();no++)
        		{
        			sum+=(double)(curNodeList.get(no).numCommNodes[c]*10+curNodeList.get(no).TotNumCommNodes);
        		}
        		for(int no=0;no<curNodeList.size();no++)
        		{
        			p[no]=(curNodeList.get(no).numCommNodes[c]*10+curNodeList.get(no).TotNumCommNodes+lambda)/(sum+lambda*curNodeList.size());	        			
        		}
        		
        		for(int pp=1; pp<curNodeList.size(); pp++)
        	        p[pp] += p[pp-1];
        	    double ran_k = random.nextDouble() * p[curNodeList.size()-1];
        	    int kk_new;
        	    for(kk_new=0; kk_new<curNodeList.size(); kk_new++){
        	      if(ran_k < p[kk_new])
        	        break;
        	    }     	    
        	    	parentNode.getChilden().get(kk_new).numCommNodes[c]++;
        	    	parentNode.getChilden().get(kk_new).TotNumCommNodes++;
        	    	curUser.setLinkTopicLevelK(link_idx, l,parentNode.getChilden().get(kk_new));

        }
        int kk_new= curUser.getLinkTopicLevel(link_idx, curUser.getLinkL(link_idx)).getID();        
        for(int v : linkContent.word_count.keySet()){//主题对应edge中每个word+其出现次数
          	try{
          		numberOfWordByTopic[kk_new][v] +=linkContent.word_count.get(v);
          		TotNumberOfWordByTopic[kk_new] += linkContent.word_count.get(v);
          	}
          	catch(Exception e)
          	{
          		 System.out.println("word Error! ");
          	}
          }	
      } 
    }

    //Sample C'
    for(int i=0; i<U; i++){
        curUser = userProfile.get(i);
        // only sample existing links
        for(int link_idx=0; link_idx < curUser.links.size(); link_idx++){
          //Link cur_link = curUser.links.get(link_idx);          
          c_ii_link_new = sampleLinkCPrime(i, link_idx);
          curUser.setLinkTargetC(link_idx, c_ii_link_new);         
        }  
      }
   
  }

  /**
   * @return the community of source node
   */
  public int sampleC(int i, int link_idx) {	 
    UserProfile curUser = userProfile.get(i);
    Link curLink=curUser.links.get(link_idx);
    //int b_ij=curLink.innerComm;
    int c_i_old = curUser.getLinkSourceC(link_idx);
    int l = curUser.getLinkL(link_idx);
    LinkedTreeNode topicNode = curUser.getLinkTopicLevel(link_idx, l);
    
    int ii=curLink.ii;
    int ii_c=curUser.getLinkTargetC(link_idx);
    numberOfCommByUser[i][c_i_old]--;
    TotNumberOfCommByUser[i]--;  
   
    numberOfTreeLevelByComm[c_i_old][l]--;
    TotNumberOfTreeLevelByComm[c_i_old]--;
    
    numberOfCommByComm[c_i_old][ii_c]--;
    TotNumberOfCommByComm[c_i_old]--;
    
    if(topicNode.getID()>0 && topicNode.getID()<4)
    {
    	numberOfLabeledTopicByComm[c_i_old][topicNode.getID()-1]--;
    	TotNumberOfLabeledTopicByComm[c_i_old]--;
    }

    int c,c_new=0;
    double[] p=new double[C];
    for(c=0; c<C; c++){
        double prior, link_alpha=1,link_delta=1;
        
        prior = (numberOfCommByUser[i][c] +rho) 
              /  (TotNumberOfCommByUser[i] + rhoSum);
        
        link_alpha=(numberOfTreeLevelByComm[c][l] + alpha) 
        /  (TotNumberOfTreeLevelByComm[c] + alphaSum);
        
        link_delta = (numberOfCommByComm[c][ii_c] + delta) 
              /  (TotNumberOfCommByComm[c] +deltaSum);

        
        p[c] = prior *  link_alpha *link_delta;
         
      }
    for(int cc=1; cc<C; cc++)
        p[cc] += p[cc-1];
    double ran = random.nextDouble() * p[C-1];
    for(c_new=0; c_new<C; c_new++){
      if(ran < p[c_new])
        break;
    }
    
   try {
    numberOfCommByUser[i][c_new]++;
   }
   catch(Exception e)
   {
	   System.out.println("Sample S Error:"+c_new);
   }
    TotNumberOfCommByUser[i]++;   
    
    numberOfTreeLevelByComm[c_new][l]++;
    TotNumberOfTreeLevelByComm[c_new]++;

    numberOfCommByComm[c_new][ii_c]++;
    TotNumberOfCommByComm[c_new]++;
    
    if(topicNode.getID()>0 && topicNode.getID()<4)
    {
    	numberOfLabeledTopicByComm[c_new][topicNode.getID()-1]++;
    	TotNumberOfLabeledTopicByComm[c_new]++;
    }
    return c_new;
  }
 
  /**
   * @return the l of source node
   */
  public int sampleL(int i, int link_idx) {	 
    UserProfile curUser = userProfile.get(i);
    Link curLink=curUser.links.get(link_idx);
    //int b_ij=curLink.innerComm;
    int l_i_old = curUser.getLinkL(link_idx);   
    int c = curUser.getLinkSourceC(link_idx);
    numberOfTreeLevelByComm[c][l_i_old]--;
    TotNumberOfTreeLevelByComm[c]--;  
    
	//int ii=curLink.ii;
    Document curDoc =curLink.linkDoc;
    int c_i_old= curUser.getLinkSourceC(link_idx);
    //int c_ii_old = curUser.getLinkTargetC(link_idx);
    int k_old = curUser.getLinkTopicLevel(link_idx, l_i_old).getID();
    
    HashMap<Integer, Integer> word_count = curDoc.word_count;
    int length = curDoc.getLength();

    for(int v : word_count.keySet()){
      int v_size = word_count.get(v);
      numberOfWordByTopic[k_old][v] -= v_size;
      TotNumberOfWordByTopic[k_old] -= v_size;
    }
   
    // probability
    int l, l_new;
    //double[] link_xi= new double[D];
    double[] link_alpha= new double[L];
    double[] p = new double[L];
    double[] logP = new double[L];
    Arrays.fill(z_post_w_carrier, 0);
	 
    for(l=0; l<L; l++){
      int k_new = curUser.getLinkTopicLevel(link_idx, l).getID();
      link_alpha[l]=(numberOfTreeLevelByComm[c][l] + alpha) 
      /  (TotNumberOfTreeLevelByComm[c] + alphaSum);
      
      z_post_w[k_new] = 1;
      z_post_w_bd[k_new] = BigDecimal.valueOf(1.0);;
      for(int v : word_count.keySet()){
        int v_size = word_count.get(v);
        for(int vcnt=0; vcnt<v_size; vcnt++){  	
          z_post_w_bd[k_new]=z_post_w_bd[k_new].multiply(BigDecimal.valueOf(numberOfWordByTopic[k_new][v] + vcnt + beta));    
           if(z_post_w_bd[k_new].doubleValue()<0)
          {
        	  System.out.println("-1");
          }
        }      
      }

      for(int vcnt=0; vcnt<length; vcnt++){       
        MathContext mc=new MathContext(2,RoundingMode.HALF_DOWN);
        z_post_w_bd[k_new]=z_post_w_bd[k_new].divide(BigDecimal.valueOf(TotNumberOfWordByTopic[k_new] + vcnt + betaSum),mc); 
      if(z_post_w_bd[k_new].doubleValue()<0)
      {
    	  System.out.println("-1");
      }   
      }
    } // end of k
    BigDecimal[] bigDp =new  BigDecimal[L];
    for(l=0; l<L; l++){    
    	int k_new = curUser.getLinkTopicLevel(link_idx, l).getID();
        bigDp[l]=BigDecimal.valueOf(link_alpha[l]).multiply(z_post_w_bd[k_new]); 
      }
    BigDecimal minP=bigDp[0];
    for(l=1; l<L; l++){
    	minP=bigDp[l].min(minP);    	
    }

    if(minP.compareTo(BigDecimal.valueOf(Double.MIN_VALUE))>-1)
    	minP=BigDecimal.valueOf(1);
    for(int ll=0; ll<L; ll++)
    {
    	p[ll]=bigDp[ll].divide(minP,325, BigDecimal.ROUND_HALF_UP).doubleValue();
    }

    // sampling
    for(int kk=1; kk<L; kk++)
      p[kk] += p[kk-1];
    double ran = random.nextDouble() * p[L-1];
    for(l_new=0; l_new<K; l_new++){
      if(ran < p[l_new])
        break;
    }
    
    for(int v : word_count.keySet()){
      int v_size = word_count.get(v);      
      numberOfWordByTopic[curUser.getLinkTopicLevel(link_idx, l_new).getID()][v] += v_size;
      TotNumberOfWordByTopic[curUser.getLinkTopicLevel(link_idx, l_new).getID()] += v_size;     
    }

    
   try {
	   numberOfTreeLevelByComm[c][l_new]++;
   }
   catch(Exception e)
   {
	   System.out.println("Sample l Error:"+l_new);
   }
   TotNumberOfTreeLevelByComm[c]++;   

    return l_new;
  }
  /**
   * @return the community of target node
   */
  public int sampleLinkCPrime(int i, int link_idx) {	 
    UserProfile curUser = userProfile.get(i);
    Link curLink=curUser.links.get(link_idx);
    int ii=curLink.ii;
    int c_i_old = curUser.getLinkSourceC(link_idx);
    int c_ii_old = curUser.getLinkTargetC(link_idx);
    int k = curUser.getLinkContentY(link_idx);
    
    //int b_ij=curLink.innerComm;
    numberOfCommByUser[ii][c_ii_old]--;
    TotNumberOfCommByUser[ii]--; 
    
    numberOfCommByComm[c_i_old][c_ii_old]--;
    TotNumberOfCommByComm[c_i_old]--;
    
    int c,c_new=0;
    double[] p=new double[C];
    for(c=0; c<C; c++){
        double  link_xi, link_eta;
        link_xi = (numberOfCommByUser[i][c] +rho) 
                /  (TotNumberOfCommByUser[i] + rhoSum);
        link_eta = (numberOfCommByComm[c_i_old][c] + delta) 
                /  (TotNumberOfCommByComm[c_i_old] +deltaSum);
          
        p[c] = link_xi *link_eta;
         
      }
    for(int cc=1; cc<C; cc++)
        p[cc] += p[cc-1];
    double ran = random.nextDouble() * p[C-1];
    for(c_new=0; c_new<C; c_new++){
      if(ran < p[c_new])
        break;
    }
    
   try {
    numberOfCommByUser[ii][c_new]++;
   }
   catch(Exception e)
   {
	   System.out.println("Sample CPrime Error:"+c_new);
   }
    TotNumberOfCommByUser[ii]++;  
    numberOfCommByComm[c_i_old][c_new]++;
    TotNumberOfCommByComm[c_i_old]++;
    return c_new;
  }
  

  
  /**
   * Method to call for fitting the model.
   * 
   * @throws IOException 
   */
  public void new_run() {
	    System.out.println("run .. ");
	    printParameter();
	    /*if(OUTPUT_LAG % SAMPLE_LAG != 0) {
	      System.err.println("OUTPUT_LAG % SAMPLE_LAG must =0");
	      System.exit(0);
	    }
	    if(LIKELIHOOD_LAG % SAMPLE_LAG != 0){
	      System.err.println("LIKELIHOOD_LAG % SAMPLE_LAG must =0");
	      System.exit(0);
	    }*/
	    
	    // initialize	  
	    initialState();
	    updateParameter();
	    //calDistribution();
	    //calLogLikelihood(0);
	    //outputLikelihood(0);
	    resetParameter();

	    // Gibbs
	   for (int iter = 1; iter <= ITERATIONS; iter++) {  
	   //for (int iter = 1; iter <= 0; iter++) {  
	      // do sampling
	    	
	      nextGibbsSweep();

	      // test
	      //if(iter==BURN_IN)
	      //{
	    	  //afterBurnInClearState();
	      //}
	      int newCurIter=iter-BURN_IN;
	      int iterLoop=0;
	      if ((iter > BURN_IN) && ((newCurIter-iterLoop*(SAMPLE_LAG+SAMPLE_PER_LAG))%SAMPLE_LAG >= 1)&&((newCurIter-iterLoop*(SAMPLE_LAG+SAMPLE_PER_LAG))%SAMPLE_LAG <= SAMPLE_PER_LAG)){
	        updateParameter();
	        calDistribution();
	        calLogLikelihood(iter);	       
	        //System.out.println("iter = " + iter);
	        iterLoop++;
//	        if(iter % OUTPUT_LAG == 0){
//	          calDistribution();
//	          outputModel(iter);
//	          outputLikelihood(iter);
//	        }
//	        if(iter % LIKELIHOOD_LAG == 0){
//	          calDistribution();
//	          calLogLikelihood(iter);
//	        }
	     /*   if(iter % OUTPUT_LAG == 0){
	        	calDistribution();
	            calLogLikelihood(iter);
	            outputModel(iter);
	            outputLikelihood(iter);
	          }*/
	         
	      }
	      System.out.println("iter = " + iter);
	    }
	   outputLikelihood();
	    //savePara(Cp,"Cpro");
	    System.out.println("Training Done.");
	  }
  
public void updateParameter(){    
   
    for(int i=0; i<U; i++){      	
      for(int c=0; c<C; c++){
    	  userCommSum[i][c] += numberOfCommByUser[i][c] ;
    	 //TotUserCommSum[i]+=userCommSum[i][c];
      }
     
    }
    // comm comm
    for (int c = 0; c < C; c++) {
    		 for (int cc = 0; cc < C; cc++) {
    			 comCommSum[c][cc]+=numberOfCommByComm[c][cc];
    		 }   	
    	}
 
    // topic word
    for (int k = 0; k < K; k++) {
		
		for (int v = 0; v < V; v++) {
			topicWordSum[k][v] += numberOfWordByTopic[k][v];
			//TotTopicWordSum[k]+= topicWordSum[k][v];
		}
	}
    

    for (int c = 0; c < C; c++) {
    	 for (int l = 0; l < L; l++) {    		 
    		 comTreeLevelSum[c][l]+=numberOfTreeLevelByComm[c][l];   		 
    	 }
    }
    
    for (int c = 0; c < C; c++) {
   	 for (int k = 0; k < 3; k++) { 
   		commLabeledTopicSum[c][k]=numberOfLabeledTopicByComm[c][k];
   		 }
   		 
   	 }
   
    sumNum++;
  }
        
  public void resetParameter(){   
   
    for(int i=0; i<U; i++){        
        Arrays.fill(userCommSum[i], 0.0); 
      }
   
    // topic word
    for(int k=0; k<K; k++){   	
      Arrays.fill(topicWordSum[k], 0.0);
    	
    }
    for (int c = 0; c < C; c++) {  
    		 Arrays.fill(comCommSum[c], 0.0); 
    		 Arrays.fill(comTreeLevelSum[c], 0.0);
    		 Arrays.fill(commLabeledTopicSum[c], 0.0);    		 
    	 }

    sumNum = 0;
  }
    
  public void calDistribution(){    
    
    // user topic - user comm
    for(int i=0; i<U; i++){
    	for (int c = 0; c < C; c++) {    	
        	userCommDistribution[i][c]= 
        			(numberOfCommByUser[i][c] + rho)/(TotNumberOfCommByUser[i] + rhoSum);
        	 }
    }

	//comm subcomm topic	
	for (int c = 0; c < C; c++) {
		for (int l = 0; l < L; l++) {
			commTreeLevelDistribution[c][l] = (numberOfTreeLevelByComm[c][l]+alpha) / (TotNumberOfTreeLevelByComm[c]+alphaSum);
		//if (userTopicDistribution[i][k] > 1.0/k)
		//	userTopic[i][k] = 1;							
	
	}
	
}
    // topic word
    for(int k=0; k<K; k++){   	
      for(int v=0; v<V; v++){
    	  topicWordDistribution[k][v] 
    			  =(numberOfWordByTopic[k][v] + beta)/(TotNumberOfWordByTopic[k] + betaSum);
        logTopicWordDistribution[k][v] = Math.log(topicWordDistribution[k][v]);
      }    
    }
    //comm comm eta
    for (int c = 0; c < C; c++) { 
    		for (int cc = 0; cc < C; cc++) {
    			comCommDistribution[c][cc]=(numberOfCommByComm[c][cc]+delta)/(TotNumberOfCommByComm[c]+deltaSum);
    		}
    }
    
    // comm labeled topic
    for (int c = 0; c < C; c++) {  	
   		 for(int i=0; i<3; i++){
   			commLabeledTopicDistribution[c][i]=(numberOfLabeledTopicByComm[c][i]+epsilon)/(TotNumberOfLabeledTopicByComm[c]+epsilon*3);
   		 }
   	 
   }		
			
  }
    
    
  public void outputParameter(String filename){
    try{
      OutputStreamWriter oswpf 
          = new OutputStreamWriter(new FileOutputStream(new File(filename)));

      oswpf.write("U\t" + U + "\n");
      oswpf.write("C\t" + C + "\n");
      oswpf.write("K\t" + K + "\n");
      oswpf.write("V\t" + V + "\n");
      //oswpf.write("T\t" + T + "\n");
      
      oswpf.write("rho\t" + rho + "\n");
      oswpf.write("alpha\t" + alpha + "\n");      
      oswpf.write("lambda\t" + lambda + "\n");
      oswpf.write("beta\t" + beta + "\n");
      oswpf.write("epsilon\t" + epsilon + "\n");
     
      oswpf.write("ITERATIONS: " + ITERATIONS + "\n");
      oswpf.write("SAMPLE_LAG: " + SAMPLE_LAG + "\n");
      oswpf.write("BURN_IN: " + BURN_IN + "\n");
      oswpf.write("outputPath: " + outputPath + "\n");
      
      oswpf.flush();
      oswpf.close();
    }
    catch(Exception e){
      e.printStackTrace();
    }
  }
    
 
    
  public void printParameter(){
    try{
        System.out.println("Markov Chain: " + numOfMarkov);
        System.out.println("Exp: " + numOfExp);
        System.out.println("----------------------------------------------");
      System.out.print("U\t" + U + "\n");
      System.out.print("C\t" + C + "\n");
      System.out.print("K\t" + K + "\n");
      System.out.print("V\t" + V + "\n");
     // System.out.print("T\t" + T + "\n");

      
      System.out.print("ITERATIONS: " + ITERATIONS + "\n");
      System.out.print("SAMPLE_LAG: " + SAMPLE_LAG + "\n");
      System.out.print("SAMPLE_PER_LAG: " + SAMPLE_PER_LAG + "\n");
      System.out.print("OUTPUT_LAG: " + OUTPUT_LAG + "\n");
      
      System.out.print("BURN_IN: " + BURN_IN + "\n");
      System.out.print("outputPath: " + outputPath + "\n");

      System.out.println("----------------------------------------------");
    }
    catch(Exception e){
      e.printStackTrace();
    }
  }
    
    
  public void calLogLikelihood(int iterNum){
    try{
      likelihood_iters.add(iterNum);

      //calDistribution();
      // total 
      double liklihood = 0.0;
      int carrier_text = 0;
      UserProfile curUser;
      int t, c, k;
      double liklihood_ij = 0;
      for(int i=0; i<U; i++){
        curUser = userProfile.get(i);
        int length = curUser.getLinkSize();
        for(int j=0; j<length; j++){
          //t = curUser.getTimestamp(j);
          c = curUser.getLinkSourceC(j);
          k = curUser.getLinkContentY(j);


          Document linkContent= curUser.links.get(j).linkDoc;
          HashMap<Integer, Integer> word_count  = linkContent.word_count;

          liklihood_ij = 0.0;
          // word
          for(int w : word_count.keySet()){
            liklihood_ij += logTopicWordDistribution[k][w] 
                    * word_count.get(w);
          }
        
          liklihood = liklihood_ij;
        } // each doc
      }
      likelihood_text.put(iterNum, liklihood);
      System.out.println(iterNum + "\t" + liklihood);
      likelihood_text_carrier.put(iterNum, carrier_text);

      // links
      double liklihood_link_tmp = 0;
      int carrier_link = 0;
      int ci, cii,ii,y,linkId;
      for(int i=0; i<U; i++){
        curUser = userProfile.get(i);
        for(int link_idx=0; link_idx < curUser.links.size(); link_idx++){
        	ci = curUser.getLinkSourceC(link_idx); 
        	cii = curUser.getLinkTargetC(link_idx);
        	Link curLink=curUser.links.get(link_idx);
        	y=curUser.getLinkContentY(link_idx);
        	ii=curLink.ii;
        	linkId=curLink.id;
        	//bij=curLink.innerComm;
          liklihood_link_tmp += Math.log(userCommDistribution[i][ci]
        		  *userCommDistribution[ii][cii]*comCommDistribution[ci][cii]);

        }
      }
      likelihood_link.put(iterNum, liklihood_link_tmp);
      System.out.println(iterNum + "\t" + liklihood_link_tmp);
      likelihood_link_carrier.put(iterNum, carrier_link);

      //comm tree level
      double liklihood_comm_tree_tmp = 0;
      int carrier_comm_tree = 0;
     
      for(int cc=0; cc<C; cc++){       
        for(int ll=0; ll < L; ll++){	
        	liklihood_comm_tree_tmp += Math.log(commTreeLevelDistribution[cc][ll]);

        }
      }
      likelihood_commTreeLevel.put(iterNum, liklihood_comm_tree_tmp);
      System.out.println(iterNum + "\t" + liklihood_comm_tree_tmp);
      likelihood_commTreeLevel_carrier.put(iterNum, carrier_comm_tree);
      
      // whole
      likelihood_final.put(iterNum, likelihood_link.get(iterNum) + likelihood_text.get(iterNum) + likelihood_commTreeLevel.get(iterNum));
      likelihood_final_carrier.put(iterNum,
          likelihood_link_carrier.get(iterNum) + likelihood_text_carrier.get(iterNum) + likelihood_commTreeLevel_carrier.get(iterNum));
      System.out.println(iterNum + "\t"
          + (likelihood_link.get(iterNum) + likelihood_text.get(iterNum)+ likelihood_commTreeLevel.get(iterNum)));
    }
    catch(Exception e){
      e.printStackTrace();
    }
  }

  public void outputLikelihood(){
    //System.out.println("output likelihood " + iter);
    try{
      String folder = outputPath + numOfExp;
      File file = new File(folder);
      if (!file.exists()) {
        file.mkdirs();
      }

      String filename = outputPath + numOfExp + "/likelihood.csv";
      OutputStreamWriter oswpf 
        = new OutputStreamWriter(new FileOutputStream(new File(filename)));
      
      for(int i=0; i<likelihood_iters.size(); i++){
        int key = likelihood_iters.get(i);
        oswpf.write(key+","+likelihood_text.get(key)+","
              +likelihood_link.get(key)+","+likelihood_final.get(key)+"\n");
      }
      oswpf.flush();
      oswpf.close();
    }
    catch(Exception e){
      e.printStackTrace();
    }
  }
}
