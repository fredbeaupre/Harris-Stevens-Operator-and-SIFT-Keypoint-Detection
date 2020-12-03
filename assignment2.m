%% QUESTION 1: GAUSSIAN SCALE SPACE
%   Compute Gaussian scale space. Split by four at each doubling
%   of octave => results in 16 slices (excluding the first one at 
%   sigma_0 = 0
%

N = 1000; % size of the image, which we'll crop later
im = ones(N,N); % initialize image
% define parameters for creating the shapes
maxsize = 200; 
circ_max = 100;
theta = 0 : 0.01 : 2*pi;

for j = 1:50 % draw 70 shapes of different types and sizes
    type = randi(4);
    % squares and rectangles
    if type == 1
        sizex= 25 + round(maxsize*rand);
        sizey = sizex;
        posx = 1+round((N-maxsize)*rand);
        posy = 1+round((N-maxsize)*rand);
        im(posy + 1:posy + sizey, posx + 1:posx + sizex) = (1 + round(255*rand));
    end
    if type == 2
        sizex = 25 + round(maxsize*rand);
        sizey = 25 + round(maxsize*rand);
        posx = 1+ round((N-maxsize)*rand);
        posy = 1+ round((N-maxsize)*rand);
        im(posy + 1:posy + sizey,posx + 1:posx + sizex) = (1+ round(255*rand));
    end
    % circles
    if type == 3
        x_center = 1 + round((N-maxsize)*rand);
        y_center = 1 + round((N-maxsize)*rand);
        radius = 5 + round((circ_max)*rand);
        graysc = (1+round(255*rand));
        for x = 1:N
            for y = 1:N
                if (x-x_center)^2 + (y - y_center)^2 <= radius^2
                    im(x,y) = graysc;
                end
            end
        end 
    end
    % ellipses
    if type == 4
        a = 200 + round((maxsize)*rand);
        b = 200 + round((maxsize)*rand);
        x_center = 1 + round((N-maxsize)*rand);
        y_center = 1 + round((N-maxsize)*rand);
        graysc = (1+round(255*rand));
        for x = 1:N
            for y = 1:N
                if (x-x_center)^2/a + (y - y_center)^2/b <= 1
                    im(x,y) = graysc;
                end
            end
        end
    end 
end
im = imcrop(im, [150 150 499 499]); % original image
im = uint8(im);
imagesc(im)
colormap(gray);
colorbar

% Uncomment below to run the assignment code with the rescaled image
% and thus check if keypoints are scale-invariant
im=imread('bobbyfisher.png');
im = rgb2gray(im);
im=imresize(im, [500 500]);
im = imresize(im, 0.6);
disp(size(im));
imagesc(im);
colormap(gray);
colorbar;

%% Now computing the scale space
factor = 2^(1/4); % splits each doubling of octave in 4 steps -> ensure log scale
sigma_0 = 1; % sigma_0
sigma_end = 16.0000; % last slice
sigma = 1;  % sigma value we'll update across scales
sigmas = ones(1, 17); % will hold the log-scaled sigma values
N = 300;
slices = zeros(N, N , 17); % array of slices
g = fspecial('gaussian', 30, sigma_0);
% compute first slice outside loop
slices(:, :, 1) = conv2(im, g, 'same');

for i=2:17
    sigmas(i) = sigmas(i-1) * factor; % update sigma
    g = fspecial('gaussian', 30, sigmas(i)); % gaussian filter
    image = conv2(im, g, 'same'); % convolve image with gaussian
    slices(:, :, i) = image; % store slice
    % plot in 4x4 grid
    subplot(4, 4, i - 1);
    imagesc(image);
    colormap(gray);
    title(['sigma = ' num2str(sigmas(i))]);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% QUESTION 2: HARRIS-STEVENS OPERATOR
%   Apply Harris-Stevens operator to each slice of the Gaussian scale
%   space computed in Question 1
%

% We use Sobel filters for computing gradients
sobel_x = [1 0 -1; 2 0 -2; 1 0 -1];
sobel_y = [1 2 1; 0 0 0; -1 -2 -1];
k = 0.1;
% loop through slices and compute gradients
for i = 2:17
    temp_image = slices(:,:, i);
    % Compute x and y derivatives (and product of derivatives
    % at every pixel)
    Ix = conv2(temp_image, sobel_x, 'same');
    Iy = conv2(temp_image, sobel_y, 'same');
    Ix2 = Ix.^2;
    Iy2 = Iy.^2;
    Ixy = Ix .* Iy;
    % Apply Gaussian 
    sigma_window = 2 * sigmas(i);
    gaussian = fspecial('gaussian', round(3*sigma_window) , sigma_window);
    M11 = conv2(Ix2, gaussian, 'same');
    M12 = conv2(Ixy, gaussian, 'same');
    M22 = conv2(Iy2, gaussian, 'same');
    % Define second moment matrix components at every pixel
    %%%%
    determinant_matrix  =  (M11 .* M22) - (M12.^2) ;
    trace_matrix = M11 + M22;
    Harris_response = determinant_matrix - (k * (trace_matrix).^2);
    subplot(4, 4, i - 1);
    imagesc(Harris_response);
    title(['sigma = ' num2str(sigmas(i))]);
    colormap jet;
    colorbar;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Question 3: DIFFERENCE OF GAUSSIANS
%   Compute difference of Gaussians scale space
%   NOTE: zero-crossings are thicker, more apparent as we go up scales
%
dog_images = zeros(N, N, 16);
for i = 2:17
    sigmas(i) = sigmas(i-1) * factor; % update sigma
    g_prev = fspecial('gaussian', 30, sigmas(i-1));
    g_curr = fspecial('gaussian', 30, sigmas(i)); % gaussian filter
    dog = g_curr - g_prev;
    dog_image = conv2(im, dog, 'same');
    dog_images(:, :, i-1) = dog_image;
    subplot(4, 4, i-1);
    imagesc(dog_image);
    colormap(gray);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% QUESTION 4: SIFT KEYPOINT DETECTION
%   Brute force search through difference of gaussians scale space to find
%   local optima
%
%
threshold = 9;
output = zeros(N, N, 16);
num_keypoints = 0;
keypoints = [];
for k = 2:15
    sigma_k = sigmas(k);
    disp(sigma_k);
    for x = round(2*sigma_k):round(N-2*sigma_k)
        for y = round(2*sigma_k):round(N-2*sigma_k)
            neighbors = (dog_images(x-1: x+1, y-1:y+1, k-1:k+1));
            neighbors = neighbors(:);
            maximum = max(neighbors, [], 'all');
            minimum = min(neighbors, [], 'all');
            output(x, y, k-1) = ( (dog_images(x, y, k) == maximum || ...
                dog_images(x,y,k) == minimum) && ...
                dog_images(x,y,k) > threshold);
            if (output(x, y, k-1)) == 1
                num_keypoints = num_keypoints + 1;
                keypoints = [keypoints; y x sigma_k k];
            end
        end
    end 
end

imagesc(im);
viscircles(keypoints(:, 1:2), 3*keypoints(:,3));
colormap(gray);

%% QUESTION 5: HESSIAN CONSTRAINT
% We use the sobel_x and sobel_y defined previously to compute
% the x and y derivatives of the entire image, i.e., dIx, dIy
% For second derivatives, we re-apply the operators on the image
% derivatives
%
blue_keypoints = [];
num_keypoints = size(keypoints, 1);
r = 10;
Lowe = ((r+1)^2)/r;
for i=1:num_keypoints
    k = keypoints(i, 4);
    x = keypoints(i, 2);
    y = keypoints(i, 1);
    temp_im = (dog_images(:,:,k));
    %first derivatives approximation
    dIx = conv2(temp_im, sobel_x, 'same');
    dIy = conv2(temp_im, sobel_y, 'same');
    % second derivatives approximation
    dIxx = conv2(dIx, sobel_x, 'same');
    dIyy = conv2(dIy, sobel_y, 'same');
    dIxy = conv2(dIx, sobel_y, 'same');
    dIyx = conv2(dIy, sobel_x, 'same');
    dIxy2 = dIxy .* dIxy;
    pt_trace = (dIxx(x,y) + dIyy(x,y))^2;
    pt_det = (dIxx(x,y) * dIyy(x,y)) - (dIxy2(x,y));
    Hessian_res = pt_trace / pt_det;
    if (Hessian_res < Lowe)
        blue_keypoints = [blue_keypoints; keypoints(i, :)];
    end  
end
imagesc(im);
viscircles(keypoints(:, 1:2), 3*keypoints(:,3));
viscircles(blue_keypoints(:,1:2), 3*blue_keypoints(:, 3), 'Color', 'b');
colormap(gray);
%% QUESTION 6: SIFT FEATURE DOMINANT ORIENTATION
%
%
% 
% 1) Compute intensity gradient at each point in scale space using local
%       differences (this time we'll use prewitt filter)
% 2) Then weight the intensity gradient images with gaussian sigma=1.5
% 3) Then define 36 bins for 360 degrees, compute orientation of the 
%       gradients. 
%        I.e: atan2(dIy/dIx) for each keypoint

prewitt_x = [1 0 -1; 1 0 -1; 1 0 -1];
prewitt_y = [1 1 1;0 0 0; -1 -1 -1];
bin_interval = 10;
binranges = -5:bin_interval:355-bin_interval;
dir_keys = [];

%dummy_data = [4 344];
%[N, edges] = histcounts(dummy_data, binranges);
%disp(N);

for i = 1:num_keypoints
    % definitions
    x = keypoints(i, 2);
    y = keypoints(i, 1);
    key_sigma = keypoints(i, 3);
    scale_id = keypoints(i, 4);
    directions = [];
    magnitudes = [];
    
    % define smoothing gaussian
    sigma_window = 1.5 * key_sigma;
    gaussian = fspecial('gaussian', round(3*sigma_window) , sigma_window);
    
    % image of interest
    temp_im = dog_images(:, :, scale_id);
    
    % compute whole image gradients, and smooth them
    dIx = conv2(temp_im, prewitt_x, 'same');
    dIy = conv2(temp_im, prewitt_y, 'same');
    dIx_smooth = conv2(dIx, gaussian, 'same');
    dIy_smooth = conv2(dIy, gaussian, 'same');
    
    % define rows and cols of interest
    rows = x-round(sigma_window):x+round(sigma_window);
    cols = y-round(sigma_window):y+round(sigma_window);
    
    for row = x-length(rows):x+length(rows)
        if row < 1 || row > 500
            continue
        end
        for col = y-length(col):y+length(cols)
            if col < 1 || col > 500
                continue
            end
            direction = atan2(dIy_smooth(row,col), dIx_smooth(row,col));
            direction = rad2deg(direction);
            direction = round(wrapTo360(direction));
            directions = [directions; direction];
        end
    end
    N = histcounts(directions, binranges);
    [dom_count, dom_ind] = max(N);
    other_doms = find(N>0.8*dom_count);
    for temp=1:length(other_doms)
        angle = other_doms(temp) * 10;
        disp(angle);
        endpoint_x = round( x + 3*sigma_k*cos(angle));
        endpoint_y = round(y + 3*sigma_k*sin(angle));
        dir_keys = [dir_keys; y x endpoint_y endpoint_x];  
    end
end

figure, imagesc(im);
viscircles(keypoints(:, 1:2), 3*keypoints(:,3));
viscircles(blue_keypoints(:,1:2), 3*blue_keypoints(:, 3), 'Color', 'b');
hold on


for i=1:length(dir_keys)
    x(1) = dir_keys(i, 1);
    x(2) = dir_keys(i, 3);
    y(1) = dir_keys(i, 2);
    y(2) = dir_keys(i,4);
    plot(x,y, 'Color', 'red');
    colormap(gray);
end

%% Histograms: Now we choose two keypoints randomly, and plot their histograms
index1 = round(rand*num_keypoints);
index2 = round(rand*num_keypoints);
keypoint1 = keypoints(index1, :);
keypoint2 = keypoints(index2, :);
x1 = keypoint1(2);
y1 = keypoint1(1);
sigma1 = keypoint1(3);
scale1 = keypoint1(4);
x2 = keypoint2(2);
y2 = keypoint2(1);
sigma2= keypoint2(3);
scale2 = keypoint2(4);

sigma_window1 = 1.5 * sigma1;
sigma_window2 = 1.5 *sigma2;
gaussian1 = fspecial('gaussian', round(3*sigma_window1), sigma_window1);
gaussian2 = fspecial('gaussian', round(3*sigma_window2), sigma_window2);

temp_im1 = dog_images(:, :, scale1);
temp_im2 = dog_images(:, : , scale2);

dIx1 = conv2(temp_im1, prewitt_x, 'same');
dIy1 = conv2(temp_im1, prewitt_y, 'same');
dIx_smooth1 = conv2(dIx1, gaussian1, 'same');
dIy_smooth1 = conv2(dIy1, gaussian1, 'same');

dIx2 = conv2(temp_im2, prewitt_x, 'same');
dIy2 = conv2(temp_im2, prewitt_y, 'same');
dIx_smooth2 = conv2(dIx2, gaussian2, 'same');
dIy_smooth2 = conv2(dIy2, gaussian2, 'same');

rows1 = x1-round(sigma_window1):x1+round(sigma_window1);
cols1 = y1-round(sigma_window1):y1+round(sigma_window1);
rows2 = x2-round(sigma_window2):x2+round(sigma_window2);
cols2 = y2-round(sigma_window2):y2+round(sigma_window2);

directions = [];
for row = x1-length(rows1):x1+length(rows1)
    if row < 1 || row > 500
        continue
    end
    for col = y1-length(cols1):y1+length(cols1)
        if col < 1 || col > 500
            continue
        end
        direction = atan2(dIy_smooth1(row,col), dIx_smooth1(row,col));
        direction = rad2deg(direction);
        direction = round(wrapTo360(direction));
        directions = [directions; direction];
    end
end

[N, edges] = histcounts(directions, binranges);
disp(N);
disp(x1);
disp(y1);
edges = edges(2:end) - (edges(2)-edges(1))/2;
plot(edges, N);


    








