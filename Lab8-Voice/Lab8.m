clear all
close all
clc

%number of samples for train and test set
NsubSet = 5; 
Nstates = 8;    
Nsamples = 2000;
Kquant = 8;
Nfiles = 10;
healty_voice = zeros(Nfiles, Nsamples);
patient_voice = zeros(Nfiles, Nsamples);
health_path = 'F:\Polito\Ict_For_Health\Lab8\data\healthy\H00';
patient_path = 'F:\Polito\Ict_For_Health\Lab8\data\parkins\P00';

%pre process the data
for i = 0 : (Nfiles - 1)
    file = strcat(health_path,int2str(i),'a1.wav');
    healty_voice (i+1, :) = pre_processing(file);
    file2 = strcat(patient_path,int2str(i),'a1.wav');
    patient_voice (i+1, :) = pre_processing(file2);   
end

healty_train_set = zeros(NsubSet, Nsamples);
healty_test_set = zeros(NsubSet, Nsamples);
patient_train_set = zeros(NsubSet, Nsamples);
patient_test_set = zeros(NsubSet, Nsamples);

%create train set
for i = 1: NsubSet
  healty_train_set(i, :) = healty_voice (i, :);
  patient_train_set(i, :) = patient_voice (i, :);
end

%create test set
for i = 1: NsubSet
  healty_test_set(i, :) = healty_voice (i+5, :);
  patient_test_set(i, :) = patient_voice (i+5, :);
end

%Training the HMM
%TRANS HAT is the initial guess of the HMM transition matrix
%EMIT HAT is the initial guess of the HMM emission matrix

% set random seed to regenerate same random numbers every time
rng(1234);

TRANS_HAT=rand(Nstates, Nstates); 
EMIT_HAT=rand(Nstates, Kquant);

%normalising
for i = 1:Nstates
  sum_TRANS = sum((TRANS_HAT(i,:)));
  sum_EMIT = sum((EMIT_HAT(i,:)));
  TRANS_HAT(i,:)=(TRANS_HAT(i,:))/sum_TRANS;  
  EMIT_HAT(i,:)=(EMIT_HAT(i,:))/sum_EMIT;
end

% training
% Baum-Welch
% ESTTR is the (output) estimated transition matrix, whereas ESTEMIT is the (output) estimated emission matrix

[ESTTR_Welch_healthy,ESTEMIT_Welch_healthy] = hmmtrain(healty_train_set,TRANS_HAT,EMIT_HAT,'Tolerance',1e-3,'Maxiterations',200);
[ESTTR_Welch_patient,ESTEMIT_Welch_patient] = hmmtrain(patient_train_set,TRANS_HAT,EMIT_HAT,'Tolerance',1e-3,'Maxiterations',200);

%Viterbi algorithm
[ESTTR_Viterbi_healthy,ESTEMIT_Viterbi_healthy] = hmmtrain(healty_train_set,TRANS_HAT,EMIT_HAT,'Tolerance',1e-3,'Maxiterations',200, 'ALGORITHM', 'Viterbi');
[ESTTR_Viterbi_patient,ESTEMIT_Viterbi_patient] = hmmtrain(patient_train_set,TRANS_HAT,EMIT_HAT,'Tolerance',1e-3,'Maxiterations',200, 'ALGORITHM', 'Viterbi');

LOGP_Welch_healty_train = zeros(1, Nfiles);
LOGP_Welch_patient_train = zeros(1, Nfiles);
LOGP_Welch_healty_test = zeros(1, Nfiles);
LOGP_Welch_patient_test = zeros(1, Nfiles);

LOGP_Viterbi_healty_train = zeros(1,Nfiles);
LOGP_Viterbi_patient_train = zeros(1, Nfiles);
LOGP_Viterbi_healty_test = zeros(1, Nfiles);
LOGP_Viterbi_patient_test = zeros(1, Nfiles);

%calculate probabilities
%[PSTATES,logpseq] = hmmdecode(...) returns logpseq, the logarithm of the probability of 
% sequence seq, given transition matrix TRANS and emission matrix EMIS.

for i = 1:NsubSet
    %Welch
    %train
    [PSTATES_Welch_healty_train,LOGP] = hmmdecode(healty_train_set(i,:),ESTTR_Welch_healthy,ESTEMIT_Welch_healthy);
    LOGP_Welch_healty_train(i) = LOGP;
    [PSTATES_Welch_healty_train,LOGP] = hmmdecode(patient_train_set(i,:),ESTTR_Welch_healthy,ESTEMIT_Welch_healthy);
    LOGP_Welch_healty_train(i+5) = LOGP;
    
    [PSTATES_Welch_patient_train,LOGP] = hmmdecode(healty_train_set(i,:),ESTTR_Welch_patient,ESTEMIT_Welch_patient);
    LOGP_Welch_patient_train(i) = LOGP;
    [PSTATES_Welch_patient_train,LOGP] = hmmdecode(patient_train_set(i,:),ESTTR_Welch_patient,ESTEMIT_Welch_patient);
    LOGP_Welch_patient_train(i+5) = LOGP;
    
    %test
    [PSTATES_Welch_healty_test,LOGP] = hmmdecode(healty_test_set(i,:),ESTTR_Welch_healthy,ESTEMIT_Welch_healthy);
    LOGP_Welch_healty_test(i) = LOGP;
    [PSTATES_Welch_healty_test,LOGP] = hmmdecode(patient_test_set(i,:),ESTTR_Welch_healthy,ESTEMIT_Welch_healthy);
    LOGP_Welch_healty_test(i+5) = LOGP;
    
    [PSTATES_Welch_patient_test,LOGP] = hmmdecode(healty_test_set(i,:),ESTTR_Welch_patient,ESTEMIT_Welch_patient);
    LOGP_Welch_patient_test(i) = LOGP;
    [PSTATES_Welch_patient_test,LOGP] = hmmdecode(patient_test_set(i,:),ESTTR_Welch_patient,ESTEMIT_Welch_patient);
    LOGP_Welch_patient_test(i+5) = LOGP;
    
    %Viterbi
    %train
    [PSTATES_Viterbi_healty_train,LOGP] = hmmdecode(healty_train_set(i,:),ESTTR_Viterbi_healthy,ESTEMIT_Viterbi_healthy);
    LOGP_Viterbi_healty_train(i) = LOGP;
    [PSTATES_Viterbi_healty_train,LOGP] = hmmdecode(patient_train_set(i,:),ESTTR_Viterbi_healthy,ESTEMIT_Viterbi_healthy);
    LOGP_Viterbi_healty_train(i+5) = LOGP;
    
    [PSTATES_Viterbi_patient_train,LOGP] = hmmdecode(healty_train_set(i,:),ESTTR_Viterbi_patient,ESTEMIT_Viterbi_patient);
    LOGP_Viterbi_patient_train(i) = LOGP;
    [PSTATES_Viterbi_patient_train,LOGP] = hmmdecode(patient_train_set(i,:),ESTTR_Viterbi_patient,ESTEMIT_Viterbi_patient);
    LOGP_Viterbi_patient_train(i+5) = LOGP;
    
    %test
    [PSTATES_Viterbi_healty_test,LOGP] = hmmdecode(healty_test_set(i,:),ESTTR_Viterbi_healthy,ESTEMIT_Viterbi_healthy);
    LOGP_Viterbi_healty_test(i) = LOGP;
    [PSTATES_Viterbi_healty_test,LOGP] = hmmdecode(patient_test_set(i,:),ESTTR_Viterbi_healthy,ESTEMIT_Viterbi_healthy);
    LOGP_Viterbi_healty_test(i+5) = LOGP;
    
    [PSTATES_Viterbi_patient_test,LOGP] = hmmdecode(healty_test_set(i,:),ESTTR_Viterbi_patient,ESTEMIT_Viterbi_patient);
    LOGP_Viterbi_patient_test(i) = LOGP;
    [PSTATES_Viterbi_patient_test,LOGP] = hmmdecode(patient_test_set(i,:),ESTTR_Viterbi_patient,ESTEMIT_Viterbi_patient);
    LOGP_Viterbi_patient_test(i+5) = LOGP;
    
end

% classification
correct_Welch_train = 0;
correct_Welch_test = 0;
correct_Viterbi_train = 0;
correct_Viterbi_test = 0;

healty_Welch_train = 0;
healty_Welch_test = 0;
healty_Viterbi_train = 0;
healty_Viterbi_test = 0;

patient_Welch_train = 0;
patient_Welch_test = 0;
patient_Viterbi_train = 0;
patient_Viterbi_test = 0;

for i = 1:NsubSet
    %Welch
    %train
    if LOGP_Welch_healty_train(i) > LOGP_Welch_patient_train(i)
        correct_Welch_train = correct_Welch_train + 1;
        healty_Welch_train = healty_Welch_train + 1;
    end
    
    if LOGP_Welch_patient_train(i + 5) > LOGP_Welch_healty_train(i + 5)
       correct_Welch_train = correct_Welch_train + 1;
       patient_Welch_train = patient_Welch_train + 1;
    end
    
    %test
    if LOGP_Welch_healty_test(i) > LOGP_Welch_patient_test(i)
        correct_Welch_test = correct_Welch_test + 1;
        healty_Welch_test = healty_Welch_test +1;
    end
    
    if LOGP_Welch_patient_test(i + 5) > LOGP_Welch_healty_test(i + 5)
        correct_Welch_test = correct_Welch_test + 1;
        patient_Welch_test = patient_Welch_test + 1;
    end
    
    %Viterbi
    %train
    if LOGP_Viterbi_healty_train(i) > LOGP_Viterbi_patient_train(i)
        correct_Viterbi_train = correct_Viterbi_train + 1;
        healty_Viterbi_train = healty_Viterbi_train + 1;
    end
    
    if LOGP_Viterbi_patient_train(i + 5) > LOGP_Viterbi_healty_train(i + 5)
       correct_Viterbi_train = correct_Viterbi_train + 1;
       patient_Viterbi_train = patient_Viterbi_train + 1;
    end
    
    %test
    if LOGP_Viterbi_healty_test(i) > LOGP_Viterbi_patient_test(i)
        correct_Viterbi_test = correct_Viterbi_test + 1;
        healty_Viterbi_test = healty_Viterbi_test + 1;
    end
    
    if LOGP_Viterbi_patient_test(i + 5) > LOGP_Viterbi_healty_test(i + 5)
        correct_Viterbi_test = correct_Viterbi_test + 1;
        patient_Viterbi_test = patient_Viterbi_test + 1;
    end
    
end
Welch_train_accuracy = correct_Welch_train / Nfiles;
Welch_test_accuracy = correct_Welch_test / Nfiles;
Viterbi_train_accuracy = correct_Viterbi_train / Nfiles;
Viterbi_test_accuracy = correct_Viterbi_test /Nfiles;
