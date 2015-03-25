package run;

import gtrf.RunLDA;


public class Run {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		gtrf.RunLDA.run_em(gtrf.RunLDA.corpus, 5, 0.2);
		
	}

}
