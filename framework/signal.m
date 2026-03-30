classdef signal
    properties
        time = []
        s = []
        fs = []
        valid = 0
    end
    
    methods
        function obj = signal(s, tdef)
            % SIGNAL Construct a signal object
            if nargin >= 1
                obj.s = s(:)';
                obj.valid = 1;
            else
                obj.s = [];
                obj.valid = 0;
                tdef = 0;
            end
            
            if nargin < 2
                return
            end
            
            if isa(tdef, 'time')
                if length(obj.s) ~= tdef.num
                    error('Mismatch between signal and time lengths.');
                end
                obj.time = tdef;
                obj.fs = obj.time.fs;
            elseif length(tdef) > 1
                obj.time = time(tdef);
                obj.fs = obj.time.fs;
            elseif (tdef < 1) && (tdef ~= 0)
                obj.time = time(struct('begin', 0, 'num', ...
                    length(obj.s), 'tstep', tdef));
                obj.fs = obj.time.fs;
            else % tdef >= 1
                obj.time = time(struct('begin', 0, 'num', ...
                    length(obj.s), 'fs', tdef));
                obj.fs = obj.time.fs;
            end
        end
    end
end