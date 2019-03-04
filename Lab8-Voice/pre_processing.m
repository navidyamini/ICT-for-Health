function ar = pre_processing(file_path)
    %pre-processing
    %step 1
    [voice, FS]=audioread(file_path);
    %step 2
    Nsub = 5;
    Nstates = 8;    
    Nsamples = 2000;
    Kquant = 8;
    % new sampling frequency
    FSa=FS/Nsub;
    Nmin=Nsub*500;
    N=Nsub*4500;
    voice=voice(Nmin:Nsub:N-Nsub);

    %step 3
    %normalizing voice vector
    voice = (voice-mean(voice(:)))/std(voice(:));
    normalized_voice_mean_square = mean (voice.^ 2);

    %step 4
    t=[0:length(voice)-1]/FSa;% time axis
    plot(t,voice),grid on

    %step 5
    % minimum time interval between two peaks (s)
    Tmin=1./200;
    [PKS,LOCS] = findpeaks(voice,FSa,'MinPeakDistance',Tmin);

    %step 6
    tt = zeros(1,(length(LOCS)-1) * Nstates );

    first_state = LOCS(1);
    second_state =  LOCS(2);
    interval = (second_state - first_state)/Nstates;
    steps = first_state;
    for i=1:Nstates
        tt(i) = steps;
        steps = steps + interval;
    end

    for i = 2:length(LOCS)-1
        steps = LOCS(i);
        interval = (LOCS(i +1) - LOCS(i)) / Nstates;
        for j = (((i-1) * Nstates)+1): i * Nstates
            tt(j) = steps;
            steps = steps + interval;
        end
    end

    %step 7
    voice1 = interp1(t, voice, tt);

    %step 8
    voice1 = voice1(1:Nsamples);
        
    %step 9
    amax = max(voice1);
    amin = min(voice1);
    %quantization interval
    delta = (amax - amin)/(Kquant - 1);
    %quantized signal
    ar = round((voice1 - amin)/ delta) + 1;   
end

