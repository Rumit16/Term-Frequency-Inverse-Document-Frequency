import java.io.IOException;
import java.util.ArrayList;
import java.util.regex.Pattern;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

public class TFIDF extends Configured implements Tool {

	private static final Logger LOG = Logger.getLogger(TFIDF.class);

	public static void main(String[] args) throws Exception {
		int res = ToolRunner.run(new TFIDF(), args);
		System.exit(res);
	}

	public int run(String[] args) throws Exception {
		int result = 0;
		Job job1 = Job.getInstance(getConf(), " wordcount ");
		job1.setJarByClass(this.getClass());

		//Setting up job1 to run Map and Reduce task to generate intermediate results
		//intermediate results will going to generate in new output file 
		Configuration conf = new Configuration();
		FileInputFormat.addInputPaths(job1, args[0]);
		FileOutputFormat.setOutputPath(job1, new Path(args[1]));
		job1.setMapperClass(Map1.class);
		job1.setReducerClass(Reduce1.class);
		job1.setOutputKeyClass(Text.class);
		job1.setOutputValueClass(IntWritable.class);

		//to get the number of files in the input folder
		FileSystem numberOfFile = FileSystem.get(job1.getConfiguration());
		FileStatus[] status = numberOfFile.listStatus(new Path(args[0]));
		conf.set("name", Integer.toString(status.length));
		
		//setting up job2 to run second Map and Reduce task
		//this job takes an intermediate file as an input 
		Job job2 = Job.getInstance(conf, " wordcount ");
		job2.setJarByClass(this.getClass());

		FileInputFormat.addInputPaths(job2, args[1]);
		FileOutputFormat.setOutputPath(job2, new Path(args[2]));
		job2.setMapperClass(Map2.class);
		job2.setReducerClass(Reduce2.class);
		job2.setOutputKeyClass(Text.class);
		job2.setOutputValueClass(Text.class);

		if (job1.waitForCompletion(true)) {
			result = job2.waitForCompletion(true) ? 0 : 1;
		}
		return result;
	}
	
	//Same Map class as DocWordCount.java 
	public static class Map1 extends Mapper<LongWritable, Text, Text, IntWritable> {
		private final static IntWritable one = new IntWritable(1);
		private Text word = new Text();

		private static final Pattern WORD_BOUNDARY = Pattern.compile("\\s*\\b\\s*");

		public void map(LongWritable offset, Text lineText, Context context) throws IOException, InterruptedException {

			String line = lineText.toString();
			Text newWord = new Text();
			String fileName = context.getInputSplit().toString();
			fileName = fileName.substring(fileName.lastIndexOf("/") + 1);
			fileName = fileName.substring(0, fileName.indexOf(":"));
			for (String word : WORD_BOUNDARY.split(line)) {
				if (word.isEmpty()) {
					continue;
				}
				newWord = new Text(word + "#####" + fileName);
				context.write(newWord, one);
			}
		}
	}

	//same reduce class as TermFrequency.java
	public static class Reduce1 extends Reducer<Text, IntWritable, Text, DoubleWritable> {
		@Override
		public void reduce(Text word, Iterable<IntWritable> counts, Context context)
				throws IOException, InterruptedException {
			int sum = 0;
			for (IntWritable count : counts) {
				sum += count.get();
			}
			context.write(word, new DoubleWritable(calcWordFreq(sum)));
		}

		private double calcWordFreq(int termFreq) {
			double result = 0.0;
			if (termFreq > 0) {
				result = 1 + Math.log10(termFreq);
			}
			return result;
		}
	}

	//Map class for the second job
	public static class Map2 extends Mapper<LongWritable, Text, Text, Text> {
		//private final static IntWritable one = new IntWritable(1);
		private Text word = new Text();

		private static final Pattern WORD_BOUNDARY = Pattern.compile("\\s*\\b\\s*");

		public void map(LongWritable offset, Text lineText, Context context) throws IOException, InterruptedException {
			String line = lineText.toString();
			
			//Modifying the text of the line to generate the required output from the map function
			line = line.replace("\t", "=");
			String key = line.substring(0, line.indexOf("#"));
			String value = line.substring(line.lastIndexOf("#") + 1);
			
			//output from the map will be like <"Hadoop", ["file1.txt=1.3010299956639813", "file2.txt=1.0"]> 
			context.write(new Text(key), new Text(value));	
		}
	}

	public static class Reduce2 extends Reducer<Text, Text, Text, DoubleWritable> {
		@Override
		public void reduce(Text word, Iterable<Text> counts, Context context) throws IOException, InterruptedException {
			
			//getting the number of files in the input folder into the size variable 
			Configuration conf = context.getConfiguration();
			Double size = Double.parseDouble(conf.get("name"));

			double count = 0.0;
			ArrayList<String> value = new ArrayList<String>();
			
			// iterate through the input value and store it in the arrayList
			// and getting the count of number of the document containing the term
			for(Text t : counts){
				String val = t.toString();
				count++;
				value.add(val);
			}
			
			// Separating the file name from the value and append it to the key 
			// and calculate the IDF for the document
			for(int i=0;i<value.size();i++){
				String fn = value.get(i);
				String file = fn.substring(0, fn.lastIndexOf("="));
				Double val = Double.parseDouble(fn.substring(fn.indexOf("=")+1));
				context.write(new Text(word + "#####" + file), new DoubleWritable(calcInverseDocFreq(size,count)*val));				
			}
		}

		// method to calculate Inverse Document Frequency
		private double calcInverseDocFreq(double totalDoc, double docContainingTerm) {
			return Math.log10((totalDoc / docContainingTerm));
		}

	}
}
