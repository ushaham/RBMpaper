function fig = create_figure_2(n_vec,mse_t,mse_rl)
   
    fig = figure;
    plot(log10(n_vec),log10(mean(mse_t,1)),'-bs');    
    hold on;
    grid on;
    plot(log10(n_vec),log10(mean(mse_rl,1)),'-rs'); 
    xlabel('$\log_{10}(n)$','interpreter','latex');
    ylabel('$\log_{10}(\mbox{MSE})$','interpreter','latex');
    xlim([log10(n_vec(1)) log10(n_vec(end))]);
    legend('Tensor','Restricted Likelihood'); %,'L');
    set(findall(fig,'-property','FontSize'),'FontSize',16);
    set(findall(fig,'-property','FontType'),'FontType','times new roman');
    %print(fig,'compare_MSE.png','-dpsc');
end