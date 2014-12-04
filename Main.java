
import java.io.PrintWriter;



public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		int numKnownSamp = 10000;
		int numAttribut = 1000;
		int numClass = 10;
		int numUnknown = 10000;
		
		int min = 0;
		int max = 10;
		int minClass = 0;
		int maxClass = numClass-1;
		
		String minMax = "";
		String knownC ="";
		String unknownC = "";
		
		PrintWriter writer;
		try {
			writer = new PrintWriter("small12345.txt", "UTF-8");
			writer.println(numKnownSamp+" "+numAttribut+" "+numClass+" "+numUnknown+" ");
			
			for(int i = 0; i < numAttribut; i++)
			{
				if(i == 0)
				{
				 minMax = min+" "+max;
				}
				else{
				 minMax = (minMax+" "+min+" "+max);
				}
				
			}
			writer.println(minMax);
			
			for(int i = 0; i< numKnownSamp; i++)
			{
				for(int j = 0; j<numAttribut;j++ )
				{
					if(j == 0)
					{
					 knownC = ((int) (Math.random()*(maxClass - minClass+1))+minClass)+
							 " "+((int) (Math.random()*(max - min+1))+min);
					}
					else{
						knownC = (knownC+" "+((int)(Math.random()*(max - min + 1))+min));
					}
				}
				writer.println(knownC);
			}
			
			int unknownIndex = 1;
			
			for(int i = 0; i< numUnknown; i++)
			{
				for(int j = 0; j<numAttribut;j++ )
				{
					if(j == 0)
					{
						unknownC = "P"+unknownIndex+
							 " "+((int) (Math.random()*(max - min+1))+min);
					}
					else{
						unknownC = (unknownC+" "+((int)(Math.random()*(max - min + 1))+min));
					}
				}
				unknownIndex++;
				writer.println(unknownC);
				
			}
			
			
			writer.close();
			System.out.println("Finished");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} 
	}

}
