clear all;

addpath '/home/thomas/tools/matlab/MIRtoolbox1.6.1/MIRToolbox'
addpath '/home/thomas/tools/matlab/AuditoryToolbox'

% addpath '/Users/tpellegrini/work/matlab/MIRToolbox'
% addpath '/Users/tpellegrini/work/matlab/AuditoryToolbox'

% datadir='/homelocal/corpora/ff1010bird';
% datadir='/baie/corpus/BAD2016/warblrb10k_public';
% datadir='/baie/corpus/BAD2016/ff1010bird';
datadir='/baie/corpus/BAD2016/bad2016test';

wavdir=[datadir, '/wav'];
% outdir=[datadir, '/fbank'];
outdir=[datadir, '/fbank_delta_deltadelta'];

% wavdir=[datadir, '/ffmpeg_augment_wav/time'];
% outdir=[datadir, '/augment_fbank'];

% nbBands = 34;
% nbBands = 43;
nbBands = 58;
output_nbBands = 56;

fMin = 50;
% fMax = 4000;
fMax = 22050;

expected_nb_of_frames = 200;
largest_nb_of_frames = -1;
longest_file = '';

wavfiles=dir([wavdir, '/*.wav']);
wavfiles=wavfiles';

for wavfile = wavfiles
 data = [];
 nom=strsplit(wavfile.name, '.');
 nom=nom{1};
 disp(nom);
 basename = nom;

 wav=strcat(wavdir, '/', wavfile.name);

 data2write = [];
 labels2write = [];
    
 outputfile = strcat(outdir, '/', basename, '_melLogSpec', int2str(output_nbBands), '.txt');
 outputfile2 = strcat(outdir, '/', basename, '_melLogSpec', int2str(output_nbBands), '.mat');
 
 if exist(outputfile2, 'file') == 2
    continue
 end
 
   
 s = miraudio(wav);

 sp1 = mirspectrum(s, 'Frame', 0.10, 's', 0.05, 's', 'Min', fMin, 'Max', fMax, 'Bands', nbBands, 'Window', 'hamming', 'NormalInput', 'Mel', 'Log');

 spData1 = mirgetdata(sp1);

  spData1 = squeeze(spData1); % removes singleton dimensions (dim == 1)
  % size(spData1)
  num_frames1 = size(spData1, 1);
  if num_frames1 > largest_nb_of_frames
      largest_nb_of_frames = num_frames1;
      longest_file = wavfile.name;
  end

  if num_frames1 < expected_nb_of_frames
       disp(['PADDING on file: ', wav, 'duration: ', int2str(num_frames1)]);
       while num_frames1 < expected_nb_of_frames
           spData1 = [spData1; spData1(end,:)];
           num_frames1 = size(spData1, 1);
       end
       disp(['   INFO: NEW Duration: ', int2str(num_frames1)]);
  %     continue;
  elseif num_frames1 > expected_nb_of_frames
       disp(['SHORTENING on file: ', wav, 'duration: ', int2str(num_frames1)]);
       spData1 = spData1(1:expected_nb_of_frames, :);
       num_frames1 = size(spData1, 1);
       disp(['   INFO: NEW Duration: ', int2str(num_frames1)]);
  end

  data(:,:,1) = single(spData1(:, 1:output_nbBands));

  % dlmwrite(outputfile{1}, data, 'precision','%.6f');

  save(outputfile2, 'data');
    
end


