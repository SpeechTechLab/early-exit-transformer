classdef time
    properties
        vec = []
        beg = 0
        num = 0
        fs = 0
    end
    
    methods
        function obj = time(varargin)
            % TIME Construct a time object
            if nargin == 1
                if isnumeric(varargin{1})
                    % time is a vector, assume uniform sampling
                    vec = reshape(varargin{1}, 1, []);
                    obj.beg = vec(1);
                    obj.num = length(vec);
                    obj.fs = 1/(vec(2) - vec(1));
                elseif isstruct(varargin{1})
                    s = varargin{1};
                    if isfield(s, 'beg')
                        obj.beg = s.beg;
                    else
                        obj.beg = s.begin;
                    end
                    if isfield(s, 'num') && isfield(s, 'fs')
                        obj.num = s.num;
                        obj.fs = s.fs;
                    elseif isfield(s, 'num') && isfield(s, 'tstep')
                        obj.num = s.num;
                        obj.fs = 1/s.tstep;
                    end
                end
            elseif nargin == 3
                obj.beg = varargin{1};
                obj.num = varargin{2};
                obj.fs = varargin{3};
            end
        end
    end
end