function box = phase_corr_match(imbw,TT)
im1 = imbw;
im2 = TT;
[r1,c1] = size(im1);
[r2,c2] = size(im2);
aaa = min([r1,c1,r2,c2]);
I1 = im1(1:aaa,1:aaa);
I2 = im2(1:aaa,1:aaa);

    SizeX = size(I1, 1);
    SizeY = size(I1, 2);
    
    FA = fftshift(fft2(I1));
    FB = fftshift(fft2(I2));
    
    % Output (FA, FB)
    
    
    
    
    % ---------------------------------------------------------------------
   
    
    
    
    % Convolve the magnitude of the FFT with a high pass filter)
    
    IA = hipass_filter(size(I1, 1),size(I1,2)).*abs(FA);  
    IB = hipass_filter(size(I2, 1),size(I2,2)).*abs(FB);  
        
    
    
        
    % Transform the high passed FFT phase to Log Polar space
    
    L1 = transformImage(IA, SizeX, SizeY, SizeX, SizeY, 'nearest', size(IA) / 2, 'valid');
    L2 = transformImage(IB, SizeX, SizeY, SizeX, SizeY, 'nearest', size(IB) / 2, 'valid');
        
    
        
    
    % Convert log polar magnitude spectrum to FFT
    
    THETA_F1 = fft2(L1);
    THETA_F2 = fft2(L2);
    
    
    
    
    % Compute cross power spectrum of F1 and F2
    
    a1 = angle(THETA_F1);
    a2 = angle(THETA_F2);

    THETA_CROSS = exp(i * (a1 - a2));
    THETA_PHASE = real(ifft2(THETA_CROSS));

    
           
    % Find the peak of the phase correlation

    THETA_SORTED = sort(THETA_PHASE(:));  % TODO speed-up, we surely don't need to sort
    
    SI = length(THETA_SORTED):-1:(length(THETA_SORTED));

    [THETA_X, THETA_Y] = find(THETA_PHASE == THETA_SORTED(SI));
    
    
    
    % Compute angle of rotation
    
    DPP = 360 / size(THETA_PHASE, 2);

    Theta = DPP * (THETA_Y - 1);
    
    % Output (Theta)
    
    
    
    
    
    % ---------------------------------------------------------------------
   
    
        
    
    % Rotate image back by theta and theta + 180
    
    R1 = imrotate(I2, -Theta, 'nearest', 'crop');  
    R2 = imrotate(I2,-(Theta + 180), 'nearest', 'crop');
    
    % Output (R1, R2)
    
    
    
    
	% ---------------------------------------------------------------------
   
     
    
    
    % Take FFT of R1
     
    R1_F2 = fftshift(fft2(R1));
     
     
     
    % Compute cross power spectrum of R1_F2 and F2
    
    a1 = angle(FA);
    a2 = angle(R1_F2);

    R1_F2_CROSS = exp(i * (a1 - a2));
    R1_F2_PHASE = real(ifft2(R1_F2_CROSS));

    % Output (R1_F2_PHASE)
     
     
    
    
    % ---------------------------------------------------------------------
   
     
    
    % Take FFT of R2
     
    R2_F2 = fftshift(fft2(R2));
     
     
     
    % Compute cross power spectrum of R2_F2 and F2
    
    a1 = angle(FA);
    a2 = angle(R2_F2);

    R2_F2_CROSS = exp(i * (a1 - a2));
    R2_F2_PHASE = real(ifft2(R2_F2_CROSS));

    % Output (R2_F2_PHASE)
    
  
    
    % ---------------------------------------------------------------------
   
    
    
    
    % Decide whether to flip 180 or -180 depending on which was the closest

    MAX_R1_F2 = max(max(R1_F2_PHASE));
    MAX_R2_F2 = max(max(R2_F2_PHASE));
    
    if (MAX_R1_F2 > MAX_R2_F2)
        
        [y, x] = find(R1_F2_PHASE == max(max(R1_F2_PHASE)));
        
        R = R1;
        
    else
        
        [y, x] = find(R2_F2_PHASE == max(max(R2_F2_PHASE)));
        
        if (Theta < 180)
            Theta = Theta + 180;
        else
            Theta = Theta - 180;
        end
        
        R = R2;
    end
    
    % Output (R, x, y)
    
    
    
    % ---------------------------------------------------------------------
   
    
    
    
    % Ensure correct translation by taking from correct edge
    
    Tx = x - 1;
    Ty = y - 1;
    
    if (x > (size(I1, 1) / 2))
        Tx = Tx - size(I1, 1);
    end
    
    if (y > (size(I1, 2) / 2))
        Ty = Ty - size(I1, 2);
    end
       
    % Output (Sx, Sy)
    
%�����������
m0 = cos(Theta*pi/180);
m1 = sin(Theta*pi/180);
m2 = -cos(Theta*pi/180)*(Tx+aaa/2)-sin(Theta*pi/180)*(Ty+aaa/2)+aaa/2;
m3 = -sin(Theta*pi/180);
m4 = cos(Theta*pi/180);
m5 = sin(Theta*pi/180)*(Tx+aaa/2)-cos(Theta*pi/180)*(Ty+aaa/2)+aaa/2;

D = [ m0 , m1 ; m3 , m4 ];
T = [ m2 ; m5];

%ͼ��ƴ��
H = [D,T;0 0 1];
invH = inv(H);

box2 = [1  size(im2,2) size(im2,2)  1 ;
        1  1           size(im2,1)  size(im2,1) ;
        1  1           1            1 ] ;
box2_ = invH * box2 ;
box2_(1,:) = box2_(1,:) ./ box2_(3,:) ;
box2_(2,:) = box2_(2,:) ./ box2_(3,:) ;

box = box2_(1:2,:);
      
    
% ---------------------------------------------------------------------
% Performs Log Polar Transform

function [r,g,b] = transformImage(A, Ar, Ac, Nrho, Ntheta, Method, Center, Shape)

% Inputs:   A       the input image
%           Nrho    the desired number of rows of transformed image
%           Ntheta  the desired number of columns of transformed image
%           Method  interpolation method (nearest,bilinear,bicubic)
%           Center  origin of input image
%           Shape   output size (full,valid)
%           Class   storage class of A

global rho;

theta = linspace(0,2*pi,Ntheta+1); theta(end) = [];

switch Shape
case 'full'
    corners = [1 1;Ar 1;Ar Ac;1 Ac];
    d = max(sqrt(sum((repmat(Center(:)',4,1)-corners).^2,2)));
case 'valid'
    d = min([Ac-Center(1) Center(1)-1 Ar-Center(2) Center(2)-1]);
end
minScale = 1;
rho = logspace(log10(minScale),log10(d),Nrho)';  % default 'base 10' logspace - play with d to change the scale of the log axis

% convert polar coordinates to cartesian coordinates and center
xx = rho*cos(theta) + Center(1);
yy = rho*sin(theta) + Center(2);

if nargout==3
  if strcmp(Method,'nearest'), % Nearest neighbor interpolation
    r=interp2(A(:,:,1),xx,yy,'nearest');
    g=interp2(A(:,:,2),xx,yy,'nearest');
    b=interp2(A(:,:,3),xx,yy,'nearest');
  elseif strcmp(Method,'bilinear'), % Linear interpolation
    r=interp2(A(:,:,1),xx,yy,'linear');
    g=interp2(A(:,:,2),xx,yy,'linear');
    b=interp2(A(:,:,3),xx,yy,'linear');
  elseif strcmp(Method,'bicubic'), % Cubic interpolation
    r=interp2(A(:,:,1),xx,yy,'cubic');
    g=interp2(A(:,:,2),xx,yy,'cubic');
    b=interp2(A(:,:,3),xx,yy,'cubic');
  else
    error(['Unknown interpolation method: ',method]);
  end
  % any pixels outside , pad with black
  mask= (xx>Ac) | (xx<1) | (yy>Ar) | (yy<1);
  r(mask)=0;
  g(mask)=0;
  b(mask)=0;
else
  if strcmp(Method,'nearest'), % Nearest neighbor interpolation
    r=interp2(A,xx,yy,'nearest');
  elseif strcmp(Method,'bilinear'), % Linear interpolation
    r=interp2(A,xx,yy,'linear');
  elseif strcmp(Method,'bicubic'), % Cubic interpolation
    r=interp2(A,xx,yy,'cubic');
  else
    error(['Unknown interpolation method: ',method]);
  end
  % any pixels outside warp, pad with black
  mask= (xx>Ac) | (xx<1) | (yy>Ar) | (yy<1);
  r(mask)=0;
end  

% ---------------------------------------------------------------------
% Returns high-pass filter

function H = hipass_filter(ht,wd)
% hi-pass filter function
% ...designed for use with Fourier-Mellin stuff
res_ht = 1 / (ht-1);
res_wd = 1 / (wd-1);

eta = cos(pi*(-0.5:res_ht:0.5));
neta = cos(pi*(-0.5:res_wd:0.5));
X = eta'*neta;

H=(1.0-X).*(2.0-X);