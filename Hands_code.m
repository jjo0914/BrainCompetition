% Hands code
tot_rep=5;
endtime=10*tot_rep; % 끝나는 시간;
steptime=1; % steptime 0.06 고정인듯함
time_num=0:steptime:endtime; % 총시간 
time_num=time_num.'; % 열행렬로
time=seconds(time_num);% 초로

n = length(time);
    %% 초기상태
    

wrist_joint = 0 * ones(n,1);
thumb_joint1  = -20 * ones(n,1);
thumb_joint2  = -90 * ones(n,1);
thumb_joint3  = -90 * ones(n,1);

index_joint1  = -70 * ones(n,1);
index_joint2  = -70 * ones(n,1);
index_joint3  = -70 * ones(n,1);

middle_joint1 = -70 * ones(n,1);
middle_joint2 = -70 * ones(n,1); %참고
middle_joint3 = -70 * ones(n,1);

ring_joint1   = -60 * ones(n,1);
ring_joint2   = -60 * ones(n,1);
ring_joint3   = -80 * ones(n,1);

small_joint1  = -70 * ones(n,1);
small_joint2  = -70 * ones(n,1);
small_joint3  = -70 * ones(n,1);

% 자세1 3초부터 5초까지
% 자세1 각도 다시볼것

% wrist_joint(idx,1)=90;
% thumb_joint1(idx,1)=-20+40;thumb_joint2(idx,1)=-90+30; thumb_joint3(idx,1)=-90+60;
% index_joint1(idx,1)=-70+60;index_joint2(idx,1)=-70+20;index_joint3(idx,1)=-70+20;
% 
% middle_joint1(idx,1)=-70+70;middle_joint2(idx,1)=-70+20;middle_joint3(idx,1)=-70+20;
% ring_joint1(idx,1)=-60+70;ring_joint2(idx,1)=-60+20;ring_joint3(idx,1)=-80+20;
% small_joint1(idx,1)=-70+70;small_joint2(idx,1)=-70+20;small_joint3(idx,1)=-70+20;
% % 쉬고
% 
% %자세2
% idx = (time_num >= 8) & (time_num <= 10); % time_num으로 따로계산
% wrist_joint(idx,1)=0;
% thumb_joint1(idx,1)=-20+70;thumb_joint2(idx,1)=-90+30; thumb_joint3(idx,1)=-90+30;
% index_joint1(idx,1)=-70+70;index_joint2(idx,1)=-70+20;index_joint3(idx,1)=-70+0;
% middle_joint1(idx,1)=-70+40;middle_joint2(idx,1)=-70+40;middle_joint3(idx,1)=-70+10;
% ring_joint1(idx,1)=-60+60;ring_joint2(idx,1)=-60+20;ring_joint3(idx,1)=-80+0;
% small_joint1(idx,1)=-70+80;small_joint2(idx,1)=-70+5;small_joint3(idx,1)=-70+10;
% % 쉬고
% 
% %자세3
% idx = (time_num >= 13) & (time_num <= 15); % time_num으로 따로계산
% wrist_joint(idx,1)=90;
% thumb_joint1(idx,1)=-20+90;thumb_joint2(idx,1)=-90+10; thumb_joint3(idx,1)=-90+10;
% index_joint1(idx,1)=-70+90;index_joint2(idx,1)=-70+80;index_joint3(idx,1)=-70+30;
% middle_joint1(idx,1)=-70+90;middle_joint2(idx,1)=-70+90;middle_joint3(idx,1)=-70+20;
% ring_joint1(idx,1)=-60+90;ring_joint2(idx,1)=-60+70;ring_joint3(idx,1)=-80+30;
% small_joint1(idx,1)=-70+90;small_joint2(idx,1)=-70+70;small_joint3(idx,1)=-70+30;
% 
% 
% %% 관절각 -> simulink
% & TCP통신

client = tcpclient("127.0.0.1", 5000); % tcpserver or tcpclient
rep=0;

while true
    try
    data=read(client,16,'single');
    % data(1) or data(2)
    idx = (time_num >= 6+10*rep) & (time_num <= 10+10*rep); % time_num으로 따로계산
        disp("반복:"+rep);
        % 받은데이터로 처리
        %살짝 보기좋게 수정..
   
         wrist_joint(idx,1)=data(1);
                %wrist_joint(idx,1)=data(1);
        thumb_joint1(idx,1)=data(2);thumb_joint2(idx,1)=data(3);thumb_joint3(idx,1)=data(4);
        index_joint1(idx,1)=data(5);index_joint2(idx,1)=data(6);index_joint3(idx,1)=data(7);
        middle_joint1(idx,1)=data(8);middle_joint2(idx,1)=data(9);middle_joint3(idx,1)=data(10);
        ring_joint1(idx,1)=data(11);ring_joint2(idx,1)=data(12);ring_joint3(idx,1)=data(13);
        small_joint1(idx,1)=data(14);small_joint2(idx,1)=data(15);small_joint3(idx,1)=data(16);



        sim_wrist_joint = timetable(time, deg2rad(wrist_joint));
        sim_thumb_joint1  = timetable(time, deg2rad(thumb_joint1));
        sim_thumb_joint2  = timetable(time, deg2rad(thumb_joint2));
        sim_thumb_joint3  = timetable(time, deg2rad(thumb_joint3));

        sim_index_joint1  = timetable(time, deg2rad(index_joint1));
        sim_index_joint2  = timetable(time, deg2rad(index_joint2));
        sim_index_joint3  = timetable(time, deg2rad(index_joint3));

        sim_middle_joint1 = timetable(time, deg2rad(middle_joint1));
        sim_middle_joint2 = timetable(time, deg2rad(middle_joint2));
        sim_middle_joint3 = timetable(time, deg2rad(middle_joint3));
% 
        sim_ring_joint1   = timetable(time, deg2rad(ring_joint1));
        sim_ring_joint2   = timetable(time, deg2rad(ring_joint2));
        sim_ring_joint3   = timetable(time, deg2rad(ring_joint3));

        sim_small_joint1  = timetable(time, deg2rad(small_joint1));
        sim_small_joint2  = timetable(time, deg2rad(small_joint2));
        sim_small_joint3  = timetable(time, deg2rad(small_joint3));
        rep=rep+1;
    catch ME 
  
       %open_system('Hand.slx') % 열려있으면OK 안열려있으면 열리게함
       %set_param('Hand','SimulationCommand','start'); % PLAY버튼
    break;
    end

    pause(0.1)
end

clear client




%% 플롯
% joint_data = [wrist_joint thumb_joint1 thumb_joint2 thumb_joint3 ...
%               index_joint1 index_joint2 index_joint3 ...
%               middle_joint1 middle_joint2 middle_joint3 ...
%               ring_joint1 ring_joint2 ring_joint3 ...
%               small_joint1 small_joint2 small_joint3];
% 
% plot(time, joint_data, 'LineWidth', 1.2);
% grid on;
% xlabel('Time');
% ylabel('Joint Angle');
% title('Hand Joint Angles');
% legend('wirst','thumb1','thumb2','thumb3', ...
%        'index1','index2','index3', ...
%        'middle1','middle2','middle3', ...
%        'ring1','ring2','ring3', ...
%        'small1','small2','small3');