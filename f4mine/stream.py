
   

    def prepare_stream(self, min_dwell, max_dwell):
        debug("Preparing Stream")
        if len(self.dwell_list) < 1:
            self.calculate_dwells()
            
        SEM = self.SEM_settings
        print(f"Structure Size: {self.structure_size}")
        print(f"Structure Size nm : {self.structure_size_nm}")
        structure_xspan = [int(SEM.field_center[0] - self.structure_size/2), int(SEM.field_center[0] + self.structure_size/2)]
        structure_yspan = [int(SEM.field_center[1] - self.structure_size/2), int(SEM.field_center[1] + self.structure_size/2)]
        print(f"X span: {structure_xspan[1] - structure_xspan[0]}")
        
        step_size_x = self.structure_size/self.binary_array.shape[0]
        step_size_y = self.structure_size/self.binary_array.shape[1]
        
        print(f"Step Size: {step_size_x}")
        
        point_xrange = range(structure_xspan[0], structure_xspan[1], int(step_size_x))
        point_yrange = range(structure_yspan[0], structure_yspan[1], int(step_size_y))


        points = np.nonzero(self.binary_array.astype(np.float16))
        indices = list(map(list, zip(*points[:2])))
        stream_list = []
        for point_number, index_coord in enumerate(indices):
            coord_x = point_xrange[index_coord[0]]
            coord_y = point_yrange[index_coord[1]]
            #print(f"X: {coord_x}, Y: {coord_y}")
            dwell = self.dwell_list[point_number][0]/10000 #convert

            if (dwell>=min_dwell) and (dwell<= max_dwell):
                stream_line = (int(dwell), coord_x, coord_y)
                print(stream_line)
                stream_list.append(stream_line)
        return stream_list



        # coords_in_slice = np.asarray(indices_in_slice)
        
        # coordinates = coords_in_slice * np.array([x_step_size, y_step_size])


    def output_stream(self, filename, sample_factor=1):
        stream_list = self.prepare_stream(min_dwell=1, max_dwell=5e5)
        numpoints = len(stream_list)
        total_dwell = 0
        with open(filename+'.str', 'w') as f:
            f.write('s16\n1\n')
            f.write(str(numpoints)+'\n')
            for k in range(0,int(numpoints), sample_factor):
                xstring = str(stream_list[k][1])
                ystring = str(stream_list[k][1])
                dwell = stream_list[k][0]
                dwellstring = str(dwell)
                linestring = dwellstring+" "+xstring+" "+ystring+" "
                if k < int(numpoints)-1:
                    f.write(linestring + '\n')
                else:
                    f.write(linestring + " "+"0")
                total_dwell += dwell
        print(total_dwell/1e7)